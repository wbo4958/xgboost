/*
 Copyright (c) 2024 by Contributors

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

import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class XGBoostEstimatorSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("Check for Spark encryption over-the-wire") {
    val originalSslConfOpt = ss.conf.getOption("spark.ssl.enabled")
    ss.conf.set("spark.ssl.enabled", true)

    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic")
    val training = smallBinaryClassificationVector

    withClue("xgboost-spark should throw an exception when spark.ssl.enabled = true but " +
      "xgboost.spark.ignoreSsl != true") {
      val thrown = intercept[Exception] {
        new XGBoostClassifier(paramMap).setNumRound(2).setNumWorkers(numWorkers).fit(training)
      }
      assert(thrown.getMessage.contains("xgboost.spark.ignoreSsl") &&
        thrown.getMessage.contains("spark.ssl.enabled"))
    }

    // Confirm that this check can be overridden.
    ss.conf.set("xgboost.spark.ignoreSsl", true)
    new XGBoostClassifier(paramMap).setNumRound(2).setNumWorkers(numWorkers).fit(training)

    originalSslConfOpt match {
      case None =>
        ss.conf.unset("spark.ssl.enabled")
      case Some(originalSslConf) =>
        ss.conf.set("spark.ssl.enabled", originalSslConf)
    }
    ss.conf.unset("xgboost.spark.ignoreSsl")
  }

  test("nthread configuration must be no larger than spark.task.cpus") {
    val training = smallBinaryClassificationVector
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "verbosity" -> "1",
      "objective" -> "binary:logistic")
    intercept[IllegalArgumentException] {
      new XGBoostClassifier(paramMap)
        .setNumWorkers(numWorkers)
        .setNumRound(2)
        .setNthread(sc.getConf.getInt("spark.task.cpus", 1) + 1)
        .fit(training)
    }
  }

  test("test preprocess dataset") {
    val dataset = ss.createDataFrame(sc.parallelize(Seq(
      (1.0, 0, 0.5, 1.0, Vectors.dense(1.0, 2.0, 3.0), "a"),
      (0.0, 2, -0.5, 0.0, Vectors.dense(0.2, 1.2, 2.0), "b"),
      (2.0, 2, -0.4, -2.1, Vectors.dense(0.5, 2.2, 1.7), "c"),
    ))).toDF("label", "group", "margin", "weight", "features", "other")

    val classifier = new XGBoostClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setBaseMarginCol("margin")
      .setWeightCol("weight")

    val (df, indices) = classifier.preprocess(dataset)
    var schema = df.schema
    assert(!schema.names.contains("group") && !schema.names.contains("other"))
    assert(indices.labelId == schema.fieldIndex("label") &&
      indices.groupId.isEmpty &&
      indices.marginId.get == schema.fieldIndex("margin") &&
      indices.weightId.get == schema.fieldIndex("weight") &&
      indices.featureId.get == schema.fieldIndex("features") &&
      indices.featureIds.isEmpty)

    classifier.setWeightCol("")
    val (df1, indices1) = classifier.preprocess(dataset)
    schema = df1.schema
    Seq("weight", "group", "other").foreach(v => assert(!schema.names.contains(v)))
    assert(indices1.labelId == schema.fieldIndex("label") &&
      indices1.groupId.isEmpty &&
      indices1.marginId.get == schema.fieldIndex("margin") &&
      indices1.weightId.isEmpty &&
      indices1.featureId.get == schema.fieldIndex("features") &&
      indices1.featureIds.isEmpty)
  }

}
