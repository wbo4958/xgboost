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

import java.io.File

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.{RANKER_OBJS, REGRESSION_OBJS}
import ml.dmlc.xgboost4j.scala.spark.params.XGBoostParams

class XGBoostRankerSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("XGBoostRanker copy") {
    val ranker = new XGBoostRanker().setNthread(2).setNumWorkers(10)
    val rankertCopied = ranker.copy(ParamMap.empty)

    assert(ranker.uid === rankertCopied.uid)
    assert(ranker.getNthread === rankertCopied.getNthread)
    assert(ranker.getNumWorkers === ranker.getNumWorkers)
  }

  test("XGBoostRankerModel copy") {
    val model = new XGBoostRankerModel("hello").setNthread(2).setNumWorkers(10)
    val modelCopied = model.copy(ParamMap.empty)
    assert(model.uid === modelCopied.uid)
    assert(model.getNthread === modelCopied.getNthread)
    assert(model.getNumWorkers === modelCopied.getNumWorkers)
  }

  test("read/write") {
    val trainDf = smallGroupVector
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
      "objective" -> "rank:map"
    )

    def check(xgboostParams: XGBoostParams[_]): Unit = {
      assert(xgboostParams.getMaxDepth === 5)
      assert(xgboostParams.getEta === 0.2)
      assert(xgboostParams.getObjective === "rank:map")
    }

    val rankerPath = new File(tempDir.toFile, "ranker").getPath
    val ranker = new XGBoostRanker(xgbParams).setNumRound(1).setGroupCol("group")
    check(ranker)
    assert(ranker.getGroupCol === "group")

    ranker.write.overwrite().save(rankerPath)
    val loadedRanker = XGBoostRanker.load(rankerPath)
    check(loadedRanker)
    assert(loadedRanker.getGroupCol === "group")

    val model = loadedRanker.fit(trainDf)
    check(model)
    assert(model.getGroupCol === "group")

    val modelPath = new File(tempDir.toFile, "model").getPath
    model.write.overwrite().save(modelPath)
    val modelLoaded = XGBoostRankerModel.load(modelPath)
    check(modelLoaded)
    assert(modelLoaded.getGroupCol === "group")
  }

  test("validate") {
    val trainDf = smallGroupVector
    val ranker = new XGBoostRanker()
    // must define group column
    intercept[IllegalArgumentException](
      ranker.validate(trainDf)
    )
    val ranker1 = new XGBoostRanker().setGroupCol("group")
    ranker1.validate(trainDf)
    assert(ranker1.getObjective === "rank:map")
  }

  test("XGBoostRankerModel transformed schema") {
    val trainDf = smallGroupVector
    val ranker = new XGBoostRanker().setNumRound(1)
    val model = ranker.fit(trainDf)
    var out = model.transform(trainDf)
    // Transform should not discard the other columns of the transforming dataframe
    Seq("label", "margin", "weight", "features").foreach { v =>
      assert(out.schema.names.contains(v))
    }
    // Ranker does not have extra columns
    Seq("rawPrediction", "probability").foreach { v =>
      assert(!out.schema.names.contains(v))
    }
    assert(out.schema.names.contains("prediction"))
    assert(out.schema.names.length === 5)
    model.setLeafPredictionCol("leaf").setContribPredictionCol("contrib")
    out = model.transform(trainDf)
    assert(out.schema.names.contains("leaf"))
    assert(out.schema.names.contains("contrib"))
  }

  test("Supported objectives") {
    val ranker = new XGBoostRanker().setGroupCol("group")
    val df = smallGroupVector
    RANKER_OBJS.foreach { obj =>
      ranker.setObjective(obj)
      ranker.validate(df)
    }

    ranker.setObjective("binary:logistic")
    intercept[IllegalArgumentException](
      ranker.validate(df)
    )
  }

  test("XGBoost-Spark output should match XGBoost4j") {
    val trainingDM = new DMatrix(Regression.train.iterator)
    val testDM = new DMatrix(Regression.test.iterator)
    val trainingDF = buildDataFrame(Regression.train)
    val testDF = buildDataFrame(Regression.test)
    val paramMap = Map("objective" -> "reg:squarederror")
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF, 5, paramMap)
  }

  test("XGBoost-Spark output with weight should match XGBoost4j") {
    val trainingDM = new DMatrix(Regression.trainWithWeight.iterator)
    trainingDM.setWeight(Regression.randomWeights)
    val testDM = new DMatrix(Regression.test.iterator)
    val trainingDF = buildDataFrame(Regression.trainWithWeight)
    val testDF = buildDataFrame(Regression.test)
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF,
      5, Map.empty, Some("weight"))
  }

  private def checkResultsWithXGBoost4j(
      trainingDM: DMatrix,
      testDM: DMatrix,
      trainingDF: DataFrame,
      testDF: DataFrame,
      round: Int = 5,
      xgbParams: Map[String, Any] = Map.empty,
      weightCol: Option[String] = None): Unit = {
    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "base_score" -> 0.5,
      "max_bin" -> 16) ++ xgbParams
    val xgb4jModel = ScalaXGBoost.train(trainingDM, paramMap, round)

    val ranker = new XGBoostRanker(paramMap)
      .setNumRound(round)
      .setNumWorkers(numWorkers)
      .setLeafPredictionCol("leaf")
      .setContribPredictionCol("contrib")
    weightCol.foreach(weight => ranker.setWeightCol(weight))

    def checkEqual(left: Array[Array[Float]], right: Map[Int, Array[Float]]) = {
      assert(left.size === right.size)
      left.zipWithIndex.foreach { case (leftValue, index) =>
        assert(leftValue.sameElements(right(index)))
      }
    }

    val xgbSparkModel = ranker.fit(trainingDF)
    val rows = xgbSparkModel.transform(testDF).collect()

    // Check Leaf
    val xgb4jLeaf = xgb4jModel.predictLeaf(testDM)
    val xgbSparkLeaf = rows.map(row =>
      (row.getAs[Int]("id"), row.getAs[DenseVector]("leaf").toArray.map(_.toFloat))).toMap
    checkEqual(xgb4jLeaf, xgbSparkLeaf)

    // Check contrib
    val xgb4jContrib = xgb4jModel.predictContrib(testDM)
    val xgbSparkContrib = rows.map(row =>
      (row.getAs[Int]("id"), row.getAs[DenseVector]("contrib").toArray.map(_.toFloat))).toMap
    checkEqual(xgb4jContrib, xgbSparkContrib)

    // Check prediction
    val xgb4jPred = xgb4jModel.predict(testDM)
    val xgbSparkPred = rows.map(row => {
      val pred = row.getAs[Double]("prediction").toFloat
      (row.getAs[Int]("id"), Array(pred))
    }).toMap
    checkEqual(xgb4jPred, xgbSparkPred)
  }

}
