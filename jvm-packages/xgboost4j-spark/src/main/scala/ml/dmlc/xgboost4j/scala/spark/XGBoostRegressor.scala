package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, MLReadable, MLReader}
import org.apache.spark.ml.xgboost.SparkUtils
import org.apache.spark.sql.Dataset

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor._uid
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.REGRESSION_OBJS

class XGBoostRegressor(override val uid: String,
                       private val xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostRegressor, XGBoostRegressionModel]
    with XGBoostEstimator[XGBoostRegressor, XGBoostRegressionModel] {

  def this() = this(_uid, Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(_uid, xgboostParams)

  xgboost2SparkParams(xgboostParams)

  /**
   * Validate the parameters before training, throw exception if possible
   */
  override protected[spark] def validate(dataset: Dataset[_]): Unit = {
    super.validate(dataset)

    // If the objective is set explicitly, it must be in binaryClassificationObjs and
    // multiClassificationObjs
    if (isSet(objective)) {
      val tmpObj = getObjective
      require(REGRESSION_OBJS.contains(tmpObj),
        s"Wrong objective for XGBoostRegressor, supported objs: ${REGRESSION_OBJS.mkString(",")}")
    }
  }

  override protected def createModel(
      booster: Booster,
      summary: XGBoostTrainingSummary): XGBoostRegressionModel = {
    new XGBoostRegressionModel(uid, booster, Option(summary))
  }
}

object XGBoostRegressor extends DefaultParamsReadable[XGBoostRegressor] {
  private val _uid = Identifiable.randomUID("xgbr")
}

class XGBoostRegressionModel private[ml](override val uid: String,
                                         val nativeBooster: Booster,
                                         val summary: Option[XGBoostTrainingSummary])
  extends PredictionModel[Vector, XGBoostRegressionModel]
    with XGBoostModel[XGBoostRegressionModel] {

  override def copy(extra: ParamMap): XGBoostRegressionModel = {
    val newModel = copyValues(new XGBoostRegressionModel(uid, nativeBooster, summary), extra)
    newModel.setParent(parent)
  }
}

object XGBoostRegressionModel extends MLReadable[XGBoostRegressionModel] {
  override def read: MLReader[XGBoostRegressionModel] = new ModelReader

  private class ModelReader extends XGBoostModelReader[XGBoostRegressionModel] {
    override def load(path: String): XGBoostRegressionModel = {
      val xgbModel = loadBooster(path)
      val meta = SparkUtils.loadMetadata(path, sc)
      val model = new XGBoostRegressionModel(meta.uid, xgbModel, None)
      meta.getAndSetParams(model)
      model
    }
  }
}
