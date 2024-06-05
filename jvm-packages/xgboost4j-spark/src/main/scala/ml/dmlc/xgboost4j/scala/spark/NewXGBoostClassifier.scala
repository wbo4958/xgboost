package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.util.DefaultParamsWritable

class NewXGBoostClassifier(override val uid: String)
  extends XGBoostEstimator[NewXGBoostClassifier, NewXGBoostClassificationModel]
  with DefaultParamsWritable {

}


class NewXGBoostClassificationModel() extends XGBoostModel[NewXGBoostClassificationModel] {

}
