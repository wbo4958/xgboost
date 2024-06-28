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
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams.{BINARY_CLASSIFICATION_OBJS, MULTICLASSIFICATION_OBJS}
import ml.dmlc.xgboost4j.scala.spark.params.XGBoostParams

class XGBoostRegressorSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {
  test("XGBoostRegressor copy") {
    val regressor = new XGBoostRegressor().setNthread(2).setNumWorkers(10)
    val regressortCopied = regressor.copy(ParamMap.empty)

    assert(regressor.uid === regressortCopied.uid)
    assert(regressor.getNthread === regressortCopied.getNthread)
    assert(regressor.getNumWorkers === regressor.getNumWorkers)
  }

  test("XGBoostRegressionModel copy") {
    val model = new XGBoostRegressionModel("hello").setNthread(2).setNumWorkers(10)
    val modelCopied = model.copy(ParamMap.empty)
    assert(model.uid === modelCopied.uid)
    assert(model.getNthread === modelCopied.getNthread)
    assert(model.getNumWorkers === modelCopied.getNumWorkers)
  }

  test("read/write") {
    val trainDf = smallBinaryClassificationVector
    val xgbParams: Map[String, Any] = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
    )

    def check(xgboostParams: XGBoostParams[_]): Unit = {
      assert(xgboostParams.getMaxDepth === 5)
      assert(xgboostParams.getEta === 0.2)
      assert(xgboostParams.getObjective === "reg:squarederror")
    }

    val classifierPath = new File(tempDir.toFile, "regressor").getPath
    val regressor = new XGBoostRegressor(xgbParams).setNumRound(1)
    check(regressor)

    regressor.write.overwrite().save(classifierPath)
    val loadedRegressor = XGBoostRegressor.load(classifierPath)
    check(loadedRegressor)

    val model = loadedRegressor.fit(trainDf)
    check(model)

    val modelPath = new File(tempDir.toFile, "model").getPath
    model.write.overwrite().save(modelPath)
    val modelLoaded = XGBoostClassificationModel.load(modelPath)
    check(modelLoaded)
  }

  test("XGBoostClassificationModel transformed schema") {
    val trainDf = smallBinaryClassificationVector
    val regressor = new XGBoostRegressor().setNumRound(1)
    val model = regressor.fit(trainDf)
    var out = model.transform(trainDf)
    // Transform should not discard the other columns of the transforming dataframe
    Seq("label", "margin", "weight", "features").foreach { v =>
      assert(out.schema.names.contains(v))
    }

    // Transform needs to add extra columns
    Seq("rawPrediction", "probability", "prediction").foreach { v =>
      assert(out.schema.names.contains(v))
    }

    out = model.transform(trainDf)

    // rawPrediction="", probability=""
    Seq("rawPrediction", "probability").foreach { v =>
      assert(!out.schema.names.contains(v))
    }

    assert(out.schema.names.contains("prediction"))

    model.setLeafPredictionCol("leaf").setContribPredictionCol("contrib")
    out = model.transform(trainDf)

    assert(out.schema.names.contains("leaf"))
    assert(out.schema.names.contains("contrib"))
  }

  test("Supported objectives") {
    val regressor = new XGBoostRegressor()
    val df = smallMultiClassificationVector
    (BINARY_CLASSIFICATION_OBJS.toSeq ++ MULTICLASSIFICATION_OBJS.toSeq).foreach { obj =>
      regressor.setObjective(obj)
      regressor.validate(df)
    }

    regressor.setObjective("reg:squaredlogerror")
    intercept[IllegalArgumentException](
      regressor.validate(df)
    )
  }

  test("Binaryclassification infer objective and num_class") {
    val trainDf = smallBinaryClassificationVector
    var regressor = new XGBoostRegressor()
    assert(regressor.getObjective === "reg:squarederror")
    assert(regressor.getNumClass === 0)
    regressor.validate(trainDf)
    assert(regressor.getObjective === "binary:logistic")
    assert(!regressor.isSet(regressor.numClass))

    // Infer objective according num class
    regressor = new XGBoostRegressor()
    regressor.setNumClass(2)
    intercept[IllegalArgumentException](
      regressor.validate(trainDf)
    )

    // Infer to num class according to num class
    regressor = new XGBoostRegressor()
    regressor.setObjective("binary:logistic")
    regressor.validate(trainDf)
    assert(regressor.getObjective === "binary:logistic")
    assert(!regressor.isSet(regressor.numClass))
  }

  test("MultiClassification infer objective and num_class") {
    val trainDf = smallMultiClassificationVector
    var regressor = new XGBoostRegressor()
    assert(regressor.getObjective === "reg:squarederror")
    assert(regressor.getNumClass === 0)
    regressor.validate(trainDf)
    assert(regressor.getObjective === "multi:softprob")
    assert(regressor.getNumClass === 3)

    // Infer to objective according to num class
    regressor = new XGBoostRegressor()
    regressor.setNumClass(3)
    regressor.validate(trainDf)
    assert(regressor.getObjective === "multi:softprob")
    assert(regressor.getNumClass === 3)

    // Infer to num class according to objective
    regressor = new XGBoostRegressor()
    regressor.setObjective("multi:softmax")
    regressor.validate(trainDf)
    assert(regressor.getObjective === "multi:softmax")
    assert(regressor.getNumClass === 3)
  }

  test("XGBoost-Spark binary classification output should match XGBoost4j") {
    val trainingDM = new DMatrix(Classification.train.iterator)
    val testDM = new DMatrix(Classification.test.iterator)
    val trainingDF = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)
    val paramMap = Map("objective" -> "binary:logistic")
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF, 5, paramMap)
  }

  test("XGBoost-Spark binary classification output with weight should match XGBoost4j") {
    val trainingDM = new DMatrix(Classification.trainWithWeight.iterator)
    trainingDM.setWeight(Classification.randomWeights)
    val testDM = new DMatrix(Classification.test.iterator)
    val trainingDF = buildDataFrame(Classification.trainWithWeight)
    val testDF = buildDataFrame(Classification.test)
    val paramMap = Map("objective" -> "binary:logistic")
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF,
      5, paramMap, Some("weight"))
  }

  Seq("multi:softprob", "multi:softmax").foreach { objective =>
    test(s"XGBoost-Spark multi classification with $objective output should match XGBoost4j") {
      val trainingDM = new DMatrix(MultiClassification.train.iterator)
      val testDM = new DMatrix(MultiClassification.test.iterator)
      val trainingDF = buildDataFrame(MultiClassification.train)
      val testDF = buildDataFrame(MultiClassification.test)
      val paramMap = Map("objective" -> "multi:softprob", "num_class" -> 6)
      checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF, 5, paramMap)
    }
  }

  test("XGBoost-Spark multi classification output with weight should match XGBoost4j") {
    val trainingDM = new DMatrix(MultiClassification.trainWithWeight.iterator)
    trainingDM.setWeight(MultiClassification.randomWeights)
    val testDM = new DMatrix(MultiClassification.test.iterator)
    val trainingDF = buildDataFrame(MultiClassification.trainWithWeight)
    val testDF = buildDataFrame(MultiClassification.test)
    val paramMap = Map("objective" -> "multi:softprob", "num_class" -> 6)
    checkResultsWithXGBoost4j(trainingDM, testDM, trainingDF, testDF, 5, paramMap, Some("weight"))
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

    val regressor = new XGBoostRegressor(paramMap)
      .setNumRound(round)
      .setNumWorkers(numWorkers)
      .setLeafPredictionCol("leaf")
      .setContribPredictionCol("contrib")
    weightCol.foreach(weight => regressor.setWeightCol(weight))

    def checkEqual(left: Array[Array[Float]], right: Map[Int, Array[Float]]) = {
      assert(left.size === right.size)
      left.zipWithIndex.foreach { case (leftValue, index) =>
        assert(leftValue.sameElements(right(index)))
      }
    }

    val xgbSparkModel = regressor.fit(trainingDF)
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

    def checkEqualForBinary(left: Array[Array[Float]], right: Map[Int, Array[Float]]) = {
      assert(left.size === right.size)
      left.zipWithIndex.foreach { case (leftValue, index) =>
        assert(leftValue.length === 1)
        assert(leftValue.length === right(index).length - 1)
        assert(leftValue(0) === right(index)(1))
      }
    }

    // Check probability
    val xgb4jProb = xgb4jModel.predict(testDM)
    val xgbSparkProb = rows.map(row =>
      (row.getAs[Int]("id"), row.getAs[DenseVector]("probability").toArray.map(_.toFloat))).toMap
    if (BINARY_CLASSIFICATION_OBJS.contains(regressor.getObjective)) {
      checkEqualForBinary(xgb4jProb, xgbSparkProb)
    } else {
      checkEqual(xgb4jProb, xgbSparkProb)
    }

    // Check rawPrediction
    val xgb4jRawPred = xgb4jModel.predict(testDM, outPutMargin = true)
    val xgbSparkRawPred = rows.map(row =>
      (row.getAs[Int]("id"), row.getAs[DenseVector]("rawPrediction").toArray.map(_.toFloat))).toMap
    if (BINARY_CLASSIFICATION_OBJS.contains(regressor.getObjective)) {
      checkEqualForBinary(xgb4jRawPred, xgbSparkRawPred)
    } else {
      checkEqual(xgb4jRawPred, xgbSparkRawPred)
    }
  }

}
