import ml.dmlc.xgboost4j.scala.Booster
import org.apache.spark.sql.Dataset

private[scala] object NewXGBoost {

  /**
   * Train a XGBoost booster on the dataset
   * @param dataset
   * @return
   */
  def train(dataset: Dataset[_]): Booster = {

  }

}
