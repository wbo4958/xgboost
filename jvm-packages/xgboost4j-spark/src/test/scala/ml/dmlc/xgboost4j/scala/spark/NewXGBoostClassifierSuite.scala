package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{array, col, lit, rand}
import org.scalatest.funsuite.AnyFunSuite

class NewXGBoostClassifierSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("test NewXGBoostClassifierSuite") {
    // Define the schema for the fake data

    val spark = ss
    //        val features = Array("feature1", "feature2", "feature3", "feature4")

    //    val df = Seq(
    //      (1.0, 0.0, 0.0, 0.0, 0.0, 30),
    //      (2.0, 3.0, 4.0, 4.0, 0.0, 31),
    //      (3.0, 4.0, 5.0, 5.0, 1.0, 32),
    //      (4.0, 5.0, 6.0, 6.0, 1.0, 33),
    //    ).toDF("feature1", "feature2", "feature3", "feature4", "label", "base_margin")

    var df = spark.read.parquet("/home/bobwang/data/iris/parquet")

    // Select the features and label columns
    val labelCol = "class"

    val features = df.schema.names.filter(_ != labelCol)

//    df = df.withColumn("base_margin", lit(20))
//      .withColumn("weight", rand(1))


    // Assemble the feature columns into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")
    val dataset = assembler.transform(df)

//    val arrayInput = df.select(array(features.map(col(_)): _*).as("features"),
//      col("label"), col("base_margin"))

    val est = new XGBoostClassifier()
      .setNumWorkers(1)
//      .setWeightCol("weight")
//      .setBaseMarginCol("base_margin")
      .setLabelCol(labelCol)
//      .setPredictionCol("")
      .setRawPredictionCol("")
      .setProbabilityCol("xxxx")
//      .setContribPredictionCol("contrb")
//      .setLeafPredictionCol("leaf")
    //    val est = new XGBoostClassifier().setLabelCol(labelCol)
    //    est.fit(arrayInput)
    val model = est.fit(dataset)
    model.transform(dataset).drop(features: _*).show(150, false)
  }

}
