package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}

class NewXGBoostClassifier(override val uid: String)
  extends XGBoostEstimator[NewXGBoostClassifier, NewXGBoostClassificationModel]
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("xgbc"))


  override def createModel(booster: Booster, metrics: Map[String, Array[Float]]):
    NewXGBoostClassificationModel = {
    new NewXGBoostClassificationModel()
  }
}


class NewXGBoostClassificationModel() extends XGBoostModel[NewXGBoostClassificationModel] {

}
