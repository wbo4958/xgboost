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

import ml.dmlc.xgboost4j.scala.spark.params.{NewHasGroupCol, SparkParams, XGBoostParams}
import ml.dmlc.xgboost4j.scala.spark.util.DataUtils.MLVectorToXGBLabeledPoint
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasRawPredictionCol
import org.apache.spark.ml.util.XGBoostSchemaUtils
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer

/**
 * Hold the column indexes used to get the column index
 */
private case class ColumnIndexes(label: String, features: String,
                                 weight: Option[String] = None,
                                 baseMargin: Option[String] = None,
                                 group: Option[String] = None)

private[spark] abstract class XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M],
  M <: XGBoostModel[M]
] extends Estimator[M] with XGBoostParams with SparkParams {

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
  private def preprocess(dataset: Dataset[_]): (Dataset[_], ColumnIndexes) = {
    // Columns to be selected for XGBoost
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    val schema = dataset.schema

    // TODO, support columnar and array.
    selectedCols.append(castToFloatIfNeeded(schema, getLabelCol))
    selectedCols.append(col(getFeaturesCol))

    val weightName = if (isDefined(weightCol) && getWeightCol.nonEmpty) {
      selectedCols.append(castToFloatIfNeeded(schema, getWeightCol))
      Some(getWeightCol)
    } else {
      None
    }

    val baseMarginName = if (isDefined(baseMarginCol) && getBaseMarginCol.nonEmpty) {
      selectedCols.append(castToFloatIfNeeded(schema, getBaseMarginCol))
      Some(getBaseMarginCol)
    } else {
      None
    }

    var groupName: Option[String] = None
    this match {
      case p: NewHasGroupCol =>
        // Cast group col to IntegerType if necessary
        if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
          selectedCols.append(castToFloatIfNeeded(schema, p.getGroupCol))
          groupName = Some(p.getGroupCol)
        }
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
    val columnIndexes = ColumnIndexes(
      getLabelCol,
      getFeaturesCol,
      weight = weightName,
      baseMargin = baseMarginName,
      group = groupName)
    (input, columnIndexes)
  }

  /**
   * Convert the dataframe to RDD
   *
   * @param dataset
   * @param columnsOrder the order of columns including weight/group/base margin ...
   * @return RDD
   */
  def toRdd(dataset: Dataset[_], columnIndexes: ColumnIndexes): RDD[Watches] = {

    // 1. to XGBLabeledPoint
    val labeledPointRDD = dataset.rdd.map {
      case row: Row =>
        val label = row.getFloat(row.fieldIndex(columnIndexes.label))
        val features = row.getAs[Vector](columnIndexes.features)
        val weight = columnIndexes.weight.map(v => row.getFloat(row.fieldIndex(v))).getOrElse(-1.0f)
        val baseMargin = columnIndexes.baseMargin.map(v =>
          row.getFloat(row.fieldIndex(v))).getOrElse(Float.NaN)
        val group = columnIndexes.group.map(v =>
          row.getFloat(row.fieldIndex(v))).getOrElse(-1.0f)

        // TODO support sparse vector.
        // TODO support array
        val values = features.toArray.map(_.toFloat)
        XGBLabeledPoint(label, values.length, null, values, weight, group.toInt, baseMargin)
    }

    labeledPointRDD.mapPartitions { iter =>
      // TODO  1. support external memory 2. rework or remove Watches
      val watches = new Watches(Array(new DMatrix(iter)), Array("train"), None)
      Iterator.single(watches)
    }
  }

  protected def createModel(booster: Booster, summary: XGBoostTrainingSummary): M

  override def fit(dataset: Dataset[_]): M = {
    val (input, columnIndexes) = preprocess(dataset)
    val rdd = toRdd(input, columnIndexes)

    val paramMap = Map(
      "num_rounds" -> 10,
      "num_workers" -> 1,
      "num_round" -> 1
    )

    val (booster, metrics) = NewXGBoost.train(
      dataset.sparkSession.sparkContext, rdd, paramMap)

    val summary = XGBoostTrainingSummary(metrics)
    createModel(booster, summary)
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

private[spark] abstract class XGBoostModel[M <: XGBoostModel[M]](
                                                                  override val uid: String,
                                                                  protected val booster: Booster,
                                                                  protected val trainingSummary: XGBoostTrainingSummary) extends Model[M]
  with XGBoostParams with SparkParams {

  override def copy(extra: ParamMap): M = defaultCopy(extra)

  def nativeBooster: Booster = booster

  def summary: XGBoostTrainingSummary = trainingSummary

  /**
   * Predict label for the given features.
   * This method is used to implement `transform()` and output [[predictionCol]].
   */
  //  def predict(features: Vector): Double

  //  def predictRaw(features: Vector): Vector

  override def transformSchema(schema: StructType): StructType = schema

  override def transform(dataset: Dataset[_]): DataFrame = {

    val spark = dataset.sparkSession
    val outputSchema = transformSchema(dataset.schema, logging = true)

    // Broadcast the booster to each executor.
    val bBooster = spark.sparkContext.broadcast(booster)

    val featureName = getFeaturesCol

    // Be careful about the order of columns
    var schema = dataset.schema

    var hasRawPredictionCol = false

    this match {
      case p: HasRawPredictionCol =>
        if (isDefined(p.rawPredictionCol) && p.getRawPredictionCol.nonEmpty) {
          schema = schema.add(
            StructField(XGBoostClassificationModel._rawPredictionCol, ArrayType(FloatType)))
          hasRawPredictionCol = true
        }
      case _ =>
    }

    var hasLeafPredictionCol = false
    if (isDefined(leafPredictionCol) && getLeafPredictionCol.nonEmpty) {
      schema = schema.add(StructField(getLeafPredictionCol, ArrayType(FloatType)))
      hasLeafPredictionCol = true
    }

    var hasContribPredictionCol = false
    if (isDefined(contribPredictionCol) && getContribPredictionCol.nonEmpty) {
      schema = schema.add(StructField(getContribPredictionCol, ArrayType(FloatType)))
      hasContribPredictionCol = true
    }


    // TODO configurable
    val inferBatchSize = 32 << 10
    val outputData = dataset.toDF().mapPartitions { rowIter =>

      rowIter.grouped(inferBatchSize).flatMap { batchRow =>
        val features = batchRow.iterator.map(row => row.getAs[Vector](
          row.fieldIndex(featureName)))

        // DMatrix used to prediction
        val dm = new DMatrix(features.map(_.asXGB))

        var tmpIter = batchRow.map(_.toSeq)

        if (hasRawPredictionCol) {
          val retIter = bBooster.value.predict(dm, outPutMargin = false)
          tmpIter = tmpIter.zip(retIter).map { case (a, b) =>
            a ++ Seq(b)
          }
        }

        if (hasLeafPredictionCol) {
          val retIter = bBooster.value.predictLeaf(dm)
          tmpIter = tmpIter.zip(retIter).map { case (a, b) =>
            a ++ Seq(b)
          }
        }

        if (hasContribPredictionCol) {
          val retIter = bBooster.value.predictContrib(dm)
          tmpIter = tmpIter.zip(retIter).map { case (a, b) =>
            a ++ Seq(b)
          }
        }
        tmpIter.map(Row.fromSeq)
      }

    }(Encoders.row(schema))

    outputData.toDF()
  }


}
