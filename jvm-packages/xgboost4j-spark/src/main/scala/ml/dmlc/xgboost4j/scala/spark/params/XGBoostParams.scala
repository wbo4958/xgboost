package ml.dmlc.xgboost4j.scala.spark.params

import org.apache.spark.ml.param.{IntParam, Param, ParamValidators, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasWeightCol}


trait HasBaseMarginCol extends Params {
  final val baseMarginCol: Param[String] = new Param[String](this, "baseMarginCol",
    "Initial prediction (aka base margin) column name.")

  /** @group getParam */
  final def getBaseMarginCol: String = $(baseMarginCol)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)
}

trait HasGroupCol extends Params {

  final val groupCol: Param[String] = new Param[String](this, "groupCol",
    "group column name for ranker.")

  final def getGroupCol: String = $(groupCol)

  def setGroupCol(value: String): this.type = set(groupCol, value)

}


private[spark] trait SparkParams extends Params
  with HasFeaturesCol with HasLabelCol with HasBaseMarginCol
  with HasWeightCol {

  final val numWorkers = new IntParam(this, "numWorkers", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))
  setDefault(numWorkers, 1)

  final def getNumWorkers: Int = $(numWorkers)

}

private[spark] trait XGBoostParams extends Params {

}
