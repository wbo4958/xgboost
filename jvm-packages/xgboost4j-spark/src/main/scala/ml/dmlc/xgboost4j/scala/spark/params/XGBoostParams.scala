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

import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._

trait HasInferenceSizeParams extends Params {
  /**
   * batch size in rows to be grouped for inference
   */
  final val inferBatchSize = new IntParam(this, "inferBatchSize", "batch size in rows " +
    "to be grouped for inference",
    ParamValidators.gtEq(1))

  /** @group getParam */
  final def getInferBatchSize: Int = $(inferBatchSize)
}

trait HasLeafPredictionCol extends Params {
  /**
   * Param for leaf prediction column name.
   *
   * @group param
   */
  final val leafPredictionCol: Param[String] = new Param[String](this, "leafPredictionCol",
    "name of the predictLeaf results")

  /** @group getParam */
  final def getLeafPredictionCol: String = $(leafPredictionCol)
}

trait HasContribPredictionCol extends Params {
  /**
   * Param for contribution prediction column name.
   *
   * @group param
   */
  final val contribPredictionCol: Param[String] = new Param[String](this, "contribPredictionCol",
    "name of the predictContrib results")

  /** @group getParam */
  final def getContribPredictionCol: String = $(contribPredictionCol)
}

trait HasBaseMarginCol extends Params {

  /**
   * Param for initial prediction (aka base margin) column name.
   *
   * @group param
   */
  final val baseMarginCol: Param[String] = new Param[String](this, "baseMarginCol",
    "Initial prediction (aka base margin) column name.")

  /** @group getParam */
  final def getBaseMarginCol: String = $(baseMarginCol)

}

trait HasGroupCol extends Params {

  final val groupCol: Param[String] = new Param[String](this, "groupCol", "group column name.")

  /** @group getParam */
  final def getGroupCol: String = $(groupCol)
}

/**
 * Trait for shared param featuresCols.
 */
trait HasFeaturesCols extends Params {
  /**
   * Param for the names of feature columns.
   *
   * @group param
   */
  final val featuresCols: StringArrayParam = new StringArrayParam(this, "featuresCols",
    "an array of feature column names.")

  /** @group getParam */
  final def getFeaturesCols: Array[String] = $(featuresCols)

  /** Check if featuresCols is valid */
  def isFeaturesColsValid: Boolean = {
    isDefined(featuresCols) && $(featuresCols) != Array.empty
  }
}

trait HasValidationIndicatorCol extends Params {

  final val validationIndicatorCol: Param[String] = new Param[String](this,
    "validationIndicatorCol", "Name of the column that indicates whether each row is for " +
      "training or for validation. False indicates training; true indicates validation.")

  final def getValidationIndicatorCol: String = $(validationIndicatorCol)
}

/**
 * XGBoost spark-specific parameters which should not be passed
 * into the xgboost library
 *
 * @tparam T should be the XGBoost estimators or models
 */
private[spark] trait SparkParams[T <: Params] extends Params
  with HasFeaturesCol with HasLabelCol with HasBaseMarginCol with HasWeightCol
  with HasPredictionCol with HasLeafPredictionCol with HasContribPredictionCol
  with HasInferenceSizeParams with HasValidationIndicatorCol {

  final val numWorkers = new IntParam(this, "numWorkers", "Number of workers used to train xgboost",
    ParamValidators.gtEq(1))

  final def getNumRound: Int = $(numRound)

  final val numRound = new IntParam(this, "numRound", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  setDefault(numRound -> 100, numWorkers -> 1, inferBatchSize -> (32 << 10))

  final def getNumWorkers: Int = $(numWorkers)

  def setNumWorkers(value: Int): T = set(numWorkers, value).asInstanceOf[T]

  def setNumRound(value: Int): T = set(numRound, value).asInstanceOf[T]

  def setFeaturesCol(value: String): T = set(featuresCol, value).asInstanceOf[T]

  def setLabelCol(value: String): T = set(labelCol, value).asInstanceOf[T]

  def setBaseMarginCol(value: String): T = set(baseMarginCol, value).asInstanceOf[T]

  def setWeightCol(value: String): T = set(weightCol, value).asInstanceOf[T]

  def setPredictionCol(value: String): T = set(predictionCol, value).asInstanceOf[T]

  def setLeafPredictionCol(value: String): T = set(leafPredictionCol, value).asInstanceOf[T]

  def setContribPredictionCol(value: String): T = set(contribPredictionCol, value).asInstanceOf[T]

  def setInferBatchSize(value: Int): T = set(inferBatchSize, value).asInstanceOf[T]

  def setValidationIndicatorCol(value: String): T =
    set(validationIndicatorCol, value).asInstanceOf[T]
}

/**
 * XGBoost classification spark-specific parameters which should not be passed
 * into the xgboost library
 *
 * @tparam T should be XGBoostClassifier or XGBoostClassificationModel
 */
private[spark] trait ClassificationParams[T <: Params] extends HasRawPredictionCol
  with HasProbabilityCol with HasThresholds {

  def setRawPredictionCol(value: String): T = set(rawPredictionCol, value).asInstanceOf[T]

  def setProbabilityCol(value: String): T = set(probabilityCol, value).asInstanceOf[T]

  def setThresholds(value: Array[Double]): T = set(thresholds, value).asInstanceOf[T]
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
private[spark] trait XGBoostParams[T <: Params] extends TreeBoosterParams
  with LearningTaskParams {

}
