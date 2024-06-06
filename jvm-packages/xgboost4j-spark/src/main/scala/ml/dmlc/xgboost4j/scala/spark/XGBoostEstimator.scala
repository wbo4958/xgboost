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
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.scala.spark.params.{NewHasGroupCol, SparkParams, XGBoostParams}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.XGBoostSchemaUtils
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable.ArrayBuffer

private[spark] abstract class XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M],
  M <: XGBoostModel[M]
] extends Estimator[M] with XGBoostParams with SparkParams {

  private val WEIGHT = "weight"
  private val BASE_MARGIN = "base_margin"
  private val GROUP = "group"

  /**
   * Pre-convert input double data to floats to align with XGBoost's internal float-based
   * operations to save memory usage.
   *
   * @param dataset the input dataset
   * @param name    which column will be casted to float if possible.
   * @return Dataset
   */
  private def castToFloatIfNeeded(dataset: Dataset[_], name: String): Dataset[_] = {
    if (dataset.schema(name) == DoubleType) {
      val meta = dataset.schema(name).metadata
      dataset.withColumn(name, col(name).as(name, meta).cast(FloatType))
    } else {
      dataset
    }
  }

  /**
   * Preprocess the dataset to meet the xgboost input requirement
   *
   * @param dataset
   * @return
   */
  private def preprocess(dataset: Dataset[_]): (Dataset[_], ArrayBuffer[String]) = {
    // Columns to be selected for XGBoost
    val selectedCols: ArrayBuffer[String] = ArrayBuffer.empty
    val extraColumnsOrder: ArrayBuffer[String] = ArrayBuffer.empty

    // TODO, support columnar and array.
    selectedCols.append(getLabelCol)
    selectedCols.append(getFeaturesCol)

    var input: Dataset[_] = castToFloatIfNeeded(dataset, getLabelCol)

    if (isDefined(weightCol) && getWeightCol.nonEmpty) {
      selectedCols.append(getWeightCol)
      extraColumnsOrder.append(WEIGHT)
      input = castToFloatIfNeeded(input, getWeightCol)
    }

    if (isDefined(baseMarginCol) && getBaseMarginCol.nonEmpty) {
      selectedCols.append(getBaseMarginCol)
      extraColumnsOrder.append(WEIGHT)
      input = castToFloatIfNeeded(input, getBaseMarginCol)
    }

    this match {
      case p: NewHasGroupCol =>
        // Cast group col to IntegerType if necessary
        if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
          val groupName = p.getGroupCol
          if (!input.schema(groupName).dataType.isInstanceOf[IntegerType]) {
            val meta = input.schema(groupName).metadata
            input = input.withColumn(groupName, col(groupName).as(groupName, meta).cast(FloatType))
          }
          extraColumnsOrder.append(GROUP)
          selectedCols.append(groupName)
        }
    }

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
    (input, extraColumnsOrder)
  }

  /**
   * Convert the dataframe to RDD
   *
   * @param dataset
   * @param columnsOrder the order of columns including weight/group/base margin ...
   * @return RDD
   */
  def toRdd(dataset: Dataset[_], columnsOrder: ArrayBuffer[String]): RDD[Watches] = {

    // 1. to XGBLabeledPoint
    val labeledPointRDD = dataset.rdd.map {
      case row@Row(label: Float, features: Vector) =>
        val (size, indices, values) = features match {
          case v: SparseVector => (v.size, v.indices, v.values.map(_.toFloat))
          case v: DenseVector => (v.size, null, v.values.map(_.toFloat))
        }

        var weight: Float = 1f
        var group: Int = -1
        var baseMargin: Float = Float.NaN
        columnsOrder.zipWithIndex foreach {
          case (name, pos) =>
            val index = pos + 2
            name match {
              case WEIGHT => weight = row.getFloat(index)
              case GROUP => group = row.getInt(index)
              case BASE_MARGIN => baseMargin = row.getFloat(index)
              case _ => throw new XGBoostError("Unsupported column " + name)
            }
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
    val (input, columnsOrder) = preprocess(dataset)
    val rdd = toRdd(input, columnsOrder)

    val paramMap = Map(
      "num_rounds" -> 10,
      "num_workers" -> 1
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
