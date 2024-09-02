package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.{Model, PipelineStage}

abstract class TestEstimator extends PipelineStage {

  def fit(): Unit
}
