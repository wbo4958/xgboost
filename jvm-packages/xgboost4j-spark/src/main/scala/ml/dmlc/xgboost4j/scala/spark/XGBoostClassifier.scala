/*
 Copyright (c) 2014-2024 by Contributors

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

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.params.ClassificationParams
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, udf}

class XGBoostClassifier(override val uid: String)
  extends XGBoostEstimator[XGBoostClassifier, XGBoostClassificationModel]
    with ClassificationParams[XGBoostClassifier] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("xgbc"))

  override protected def createModel(booster: Booster, summary: XGBoostTrainingSummary):
  XGBoostClassificationModel = {
    new XGBoostClassificationModel(uid, booster, summary)
  }
}


class XGBoostClassificationModel(
                                  uid: String,
                                  booster: Booster,
                                  trainingSummary: XGBoostTrainingSummary
                                )
  extends XGBoostModel[XGBoostClassificationModel](uid, booster, trainingSummary)
    with ClassificationParams[XGBoostClassificationModel] {

  override def postTranform(dataset: Dataset[_]): Dataset[_] = {
    var output = dataset
    if (isDefined(probabilityCol) && getProbabilityCol.nonEmpty) {
      output = output.withColumnRenamed(TMP_TRANSFORMED_COL, getProbabilityCol)
    }
    output
  }
}
