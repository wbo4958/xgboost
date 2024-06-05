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

import ml.dmlc.xgboost4j.scala.spark.params.{HasGroupCol, SparkParams, XGBoostParams}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.XGBoostSchemaUtils
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{DoubleType, FloatType, StructType}

import scala.collection.mutable.ArrayBuffer

private[spark] abstract class XGBoostEstimator[
  Learner <: XGBoostEstimator[Learner, M],
  M <: XGBoostModel[M]
] extends Estimator with XGBoostParams with SparkParams {

  /**
   * Pre-convert input double data to floats to align with XGBoost's internal float-based
   * operations to save memory usage.
   *
   * @param dataset the input dataset
   * @param name which column will be casted to float if possible.
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

  private def preprocess(dataset: Dataset[_]): Dataset[_] = {
    // Columns to be selected for XGBoost
    val selectedCols: ArrayBuffer[String] = ArrayBuffer.empty

    // TODO, support columnar and array.
    selectedCols.append(getFeaturesCol)
    selectedCols.append(getLabelCol)

    var input: Dataset[_] = castToFloatIfNeeded(dataset, getLabelCol)

    if (isDefined(weightCol) && getWeightCol.nonEmpty) {
      selectedCols.append(getWeightCol)
      input = castToFloatIfNeeded(input, getWeightCol)
    }

    if (isDefined(baseMarginCol) && getBaseMarginCol.nonEmpty) {
      selectedCols.append(getBaseMarginCol)
      input = castToFloatIfNeeded(input, getBaseMarginCol)
    }

    this match {
      case p: HasGroupCol =>
        if (isDefined(p.groupCol) && $(p.groupCol).nonEmpty) {
          selectedCols.append(p.getGroupCol)
          input = castToFloatIfNeeded(input, p.getGroupCol)
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
    input
  }

  override def fit(dataset: Dataset[_]): M = {
    val input = preprocess(dataset)

  }

  override def copy(extra: ParamMap): Learner = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    XGBoostSchemaUtils.checkNumericType(schema, $(labelCol))
    if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      XGBoostSchemaUtils.checkNumericType(schema, $(weightCol))
    }
    schema
  }

}

private[spark] abstract class XGBoostModel[M <: XGBoostModel[M]]
  extends Model[M] with XGBoostParams with SparkParams {
  override def copy(extra: ParamMap): M = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {

  }

  override def transformSchema(schema: StructType): StructType = schema

  override val uid: String = ???
}
