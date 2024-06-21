package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.sql.SparkSession

import ml.dmlc.xgboost4j.scala.rapids.spark.GpuTestSuite

class GpuXGBoostPluginSuite extends GpuTestSuite {

  test("isEnabled") {
    def checkIsEnabled(spark: SparkSession, expected: Boolean): Unit = {
      import spark.implicits._
      val df = Seq((1.0f, 2.0f, 0.0f),
        (2.0f, 3.0f, 1.0f)
      ).toDF("c1", "c2", "label")
      val classifier = new XGBoostClassifier()
      assert(classifier.getPlugin.isDefined)
      assert(classifier.getPlugin.get.isEnabled(df) === expected)
    }

    withCpuSparkSession() { spark =>
      checkIsEnabled(spark, false)
    }

    withGpuSparkSession() {spark =>
      checkIsEnabled(spark, true)
    }
  }

  test("preprocess") {
    withGpuSparkSession() {spark =>
    }
  }

}
