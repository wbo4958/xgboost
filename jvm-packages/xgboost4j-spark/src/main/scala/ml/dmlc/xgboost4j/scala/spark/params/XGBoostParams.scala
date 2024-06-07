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


trait NewHasBaseMarginCol extends Params {
  final val baseMarginCol: Param[String] = new Param[String](this, "baseMarginCol",
    "Initial prediction (aka base margin) column name.")

  /** @group getParam */
  final def getBaseMarginCol: String = $(baseMarginCol)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)
}

trait NewHasGroupCol extends Params {

  final val groupCol: Param[String] = new Param[String](this, "groupCol",
    "group column name for ranker.")

  final def getGroupCol: String = $(groupCol)

  def setGroupCol(value: String): this.type = set(groupCol, value)

}


private[spark] trait SparkParams extends Params
  with HasFeaturesCol with HasLabelCol with NewHasBaseMarginCol
  with HasWeightCol with HasPredictionCol with HasLeafPredictionCol with HasContribPredictionCol {

  final val numWorkers = new IntParam(this, "numWorkers", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))
  setDefault(numWorkers, 1)

  final def getNumWorkers: Int = $(numWorkers)

  def setNumWorkers(value: Int): this.type = set(numWorkers, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setLeafPredictionCol(value: String): this.type = set(leafPredictionCol, value)

  def setContribPredictionCol(value: String): this.type = set(contribPredictionCol, value)
}

private[spark] trait ClassifierParams extends HasRawPredictionCol with HasProbabilityCol {
  def setRawPredictionCol(value: String): this.type = set(rawPredictionCol, value)

  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
}

private[spark] trait XGBoostParams extends Params {

}
