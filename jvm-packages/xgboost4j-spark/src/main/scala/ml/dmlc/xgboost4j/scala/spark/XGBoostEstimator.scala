/*
 Copyright (c) 2024 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.spark.params.{NewHasGroupCol, SparkParams, XGBoostParams}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.XGBoostSchemaUtils
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.{FloatType, StructType}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.collection.mutable.ArrayBuffer

private[spark] abstract class XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M],
  M <: XGBoostModel[M]
] extends Estimator[M] with XGBoostParams with SparkParams {

  private lazy val defaultBaseMarginColumn = lit(Float.NaN)
  private lazy val defaultWeightColumn = lit(1.0f)
  private lazy val defaultGroupColumn = lit(-1.0f)

  /**
   * Pre-convert input double data to floats to align with XGBoost's internal float-based
   * operations to save memory usage.
   *
   * @param dataset the input dataset
   * @param name    which column will be casted to float if possible.
   * @return Dataset
   */
  private def castToFloatIfNeeded(schema: StructType, name: String): Column = {
    if (!schema(name).dataType.isInstanceOf[FloatType]) {
      val meta = schema(name).metadata
      col(name).as(name, meta).cast(FloatType)
    } else {
      col(name)
    }
  }

  /**
   * Preprocess the dataset to meet the xgboost input requirement
   *
   * @param dataset
   * @return
   */
  private def preprocess(dataset: Dataset[_]): (Dataset[_], Boolean, Boolean) = {
    // Columns to be selected for XGBoost
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    var hasGroup = false
    // TODO support array type
    var featureIsVector = true

    val schema = dataset.schema
    // TODO, support columnar and array.
    selectedCols.append(castToFloatIfNeeded(schema, getLabelCol))
    selectedCols.append(col(getFeaturesCol))

    val weight = if (isDefined(weightCol) && getWeightCol.nonEmpty) {
      castToFloatIfNeeded(schema, getWeightCol)
    } else {
      defaultWeightColumn
    }
    selectedCols.append(weight)

    val baseMargin = if (isDefined(baseMarginCol) && getBaseMarginCol.nonEmpty) {
      castToFloatIfNeeded(schema, getBaseMarginCol)
    } else {
      defaultBaseMarginColumn
    }
    selectedCols.append(baseMargin)


    this match {
      case p: NewHasGroupCol =>
        // Cast group col to IntegerType if necessary
        val group = if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
          castToFloatIfNeeded(schema, p.getGroupCol)
        } else {
          defaultGroupColumn
        }
        hasGroup = true
        selectedCols.append(group)

      case _ =>
    }

    var input = dataset.select(selectedCols: _*)

    // TODO,
    //  1. add a parameter to force repartition,
    //  2. follow xgboost pyspark way check if repartition is needed.
    val numWorkers = getNumWorkers
    val numPartitions = dataset.rdd.getNumPartitions
    input = if (numWorkers == numPartitions) {
      input
    } else {
      input.repartition(numWorkers)
    }
    (input, hasGroup, featureIsVector)
  }

  /**
   * Convert the dataframe to RDD
   *
   * @param dataset
   * @param columnsOrder the order of columns including weight/group/base margin ...
   * @return RDD
   */
  def toRdd(dataset: Dataset[_], hasGroup: Boolean, featureIsVector: Boolean): RDD[Watches] = {

    // 1. to XGBLabeledPoint
    val labeledPointRDD = dataset.rdd.map {
      case row: Row =>
        val label = row.getFloat(0)
        val features = row.getAs[Vector](1)
        val weight = row.getFloat(2)
        val baseMargin = row.getFloat(3)
        val group = if (hasGroup) {
          row.getInt(4)
        } else {
          -1
        }
        val (size, indices, values) = features match {
          case v: SparseVector => (v.size, v.indices, v.values.map(_.toFloat))
          case v: DenseVector => (v.size, null, v.values.map(_.toFloat))
        }
        XGBLabeledPoint(label, size, indices, values, weight, group, baseMargin)
    }

    labeledPointRDD.mapPartitions { iter =>
      // TODO  1. support external memory 2. rework or remove Watches
      val watches = new Watches(Array(new DMatrix(iter)), Array("training"), None)
      Iterator.single(watches)
    }
  }

  def createModel(booster: Booster, metrics: Map[String, Array[Float]]): M

  override def fit(dataset: Dataset[_]): M = {
    val (input, hasGroup, featureIsVector) = preprocess(dataset)
    val rdd = toRdd(input, hasGroup, featureIsVector )

    val paramMap = Map(
      "num_rounds" -> 10,
      "num_workers" -> 1,
      "num_round" -> 1
    )

    val (booster, metrics) = NewXGBoost.train(
      dataset.sparkSession.sparkContext, rdd, paramMap)

    createModel(booster, metrics)
  }

  override def copy(extra: ParamMap): Learner = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    XGBoostSchemaUtils.checkNumericType(schema, $(labelCol))
    if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      XGBoostSchemaUtils.checkNumericType(schema, $(weightCol))
    }
    this match {
      case p: NewHasGroupCol =>
        if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
          XGBoostSchemaUtils.checkNumericType(schema, p.getGroupCol)
        }
    }
    schema
  }

}

private[spark] abstract class XGBoostModel[M <: XGBoostModel[M]]
  extends Model[M] with XGBoostParams with SparkParams {
  override def copy(extra: ParamMap): M = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.asInstanceOf[DataFrame]
  }

  override def transformSchema(schema: StructType): StructType = schema

  override val uid: String = "1"
}
