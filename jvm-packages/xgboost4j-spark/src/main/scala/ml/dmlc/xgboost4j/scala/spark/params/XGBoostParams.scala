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

package ml.dmlc.xgboost4j.scala.spark.params

import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{IntParam, Param, ParamValidators, Params}

/**
 * XGBoost spark-specific parameters which should not be passed
 * into the xgboost library
 *
 * @tparam T should be the XGBoost estimators or models
 */
private[spark] trait SparkParams[T <: Params] extends Params
  with HasFeaturesCol with HasLabelCol with HasBaseMarginCol with HasWeightCol
  with HasPredictionCol with HasLeafPredictionCol with HasContribPredictionCol {

  final val numWorkers = new IntParam(this, "numWorkers", "Number of workers used to train xgboost",
    ParamValidators.gtEq(1))
  setDefault(numWorkers, 1)

  final def getNumWorkers: Int = $(numWorkers)

  def setNumWorkers(value: Int): T = set(numWorkers, value).asInstanceOf[T]

  def setFeaturesCol(value: String): T = set(featuresCol, value).asInstanceOf[T]

  def setLabelCol(value: String): T = set(labelCol, value).asInstanceOf[T]

  def setWeightCol(value: String): T = set(weightCol, value).asInstanceOf[T]

  def setLeafPredictionCol(value: String): T = set(leafPredictionCol, value).asInstanceOf[T]

  def setContribPredictionCol(value: String): T = set(contribPredictionCol, value).asInstanceOf[T]
}

/**
 * XGBoost classification spark-specific parameters which should not be passed
 * into the xgboost library
 *
 * @tparam T should be XGBoostClassifier or XGBoostClassificationModel
 */
private[spark] trait ClassificationParams[T <: Params] extends HasRawPredictionCol
  with HasProbabilityCol {

  def setRawPredictionCol(value: String): T = set(rawPredictionCol, value).asInstanceOf[T]

  def setProbabilityCol(value: String): T = set(probabilityCol, value).asInstanceOf[T]
}

/**
 * XGBoost ranking spark-specific parameters
 *
 * @tparam T should be XGBoostRanker or XGBoostRankingModel
 */
private[spark] trait RankerParams[T <: Params] extends HasGroupCol {
  def setGroupCol(value: String): T = set(groupCol, value).asInstanceOf[T]
}

/**
 * XGBoost-specific parameters to pass into xgboost libraray
 *
 * @tparam T should be the XGBoost estimators or models
 */
private[spark] trait XGBoostParams[T <: Params] extends Params {

}
