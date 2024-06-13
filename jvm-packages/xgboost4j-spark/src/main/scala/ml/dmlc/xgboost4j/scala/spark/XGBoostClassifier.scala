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
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.mutable

class XGBoostClassifier(override val uid: String,
                        private[spark] val xgboostParams: Map[String, Any])
  extends XGBoostEstimator[XGBoostClassifier, XGBoostClassificationModel]
    with ClassificationParams[XGBoostClassifier] with DefaultParamsWritable {

  private val XGBC_UID = Identifiable.randomUID("xgbc")

  def this() = this(Identifiable.randomUID("xgbc"), Map.empty)

  def this(uid: String) = this(uid, Map.empty)

  def this(xgboostParams: Map[String, Any]) = this(Identifiable.randomUID("xgbc"), xgboostParams)

  xgboost2SparkParams(xgboostParams)

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

  // Copied from Spark
  private def probability2prediction(probability: Vector): Double = {
    if (!isDefined(thresholds)) {
      probability.argmax
    } else {
      val thresholds = getThresholds
      var argMax = 0
      var max = Double.NegativeInfinity
      var i = 0
      val probabilitySize = probability.size
      while (i < probabilitySize) {
        // Thresholds are all > 0, excepting that at most one may be 0.
        // The single class whose threshold is 0, if any, will always be predicted
        // ('scaled' = +Infinity). However in the case that this class also has
        // 0 probability, the class will not be selected ('scaled' is NaN).
        val scaled = probability(i) / thresholds(i)
        if (scaled > max) {
          max = scaled
          argMax = i
        }
        i += 1
      }
      argMax
    }
  }

  override def postTransform(dataset: Dataset[_]): Dataset[_] = {
    var output = dataset
    if (isDefined(predictionCol) && getPredictionCol.nonEmpty) {
      val predCol = udf { probability: mutable.WrappedArray[Float] =>
        probability2prediction(Vectors.dense(probability.map(_.toDouble).toArray))
      }
      output = output.withColumn(getPredictionCol, predCol(col(TMP_TRANSFORMED_COL)))
    }

    if (isDefined(probabilityCol) && getProbabilityCol.nonEmpty) {
      output = output.withColumnRenamed(TMP_TRANSFORMED_COL, getProbabilityCol)
    }
    output.drop(TMP_TRANSFORMED_COL)
  }
}
