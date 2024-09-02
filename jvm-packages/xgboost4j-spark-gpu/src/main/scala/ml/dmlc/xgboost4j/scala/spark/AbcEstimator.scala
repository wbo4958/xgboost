package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.types.StructType

class AbcEstimator extends TestEstimator {

  override def fit(): Unit = {
    println("In AbcEstimator fit")
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): PipelineStage = this

  override val uid: String = "102"
}
