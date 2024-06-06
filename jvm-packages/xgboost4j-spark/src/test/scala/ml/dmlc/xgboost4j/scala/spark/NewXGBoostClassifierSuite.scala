package ml.dmlc.xgboost4j.scala.spark

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{array, col}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.scalatest.funsuite.AnyFunSuite

class NewXGBoostClassifierSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("test NewXGBoostClassifierSuite") {
    // Define the schema for the fake data
    val schema = StructType(
      StructField("feature1", DoubleType, nullable = false) ::
        StructField("feature2", DoubleType, nullable = false) ::
        StructField("feature3", DoubleType, nullable = false) ::
        StructField("label", DoubleType, nullable = false) :: Nil
    )
    val features = Array("feature1", "feature2", "feature3")

    val spark = ss
    import spark.implicits._
    val df = Seq(
      (1.0, 2.0, 3.0, 0.0, 30),
      (2.0, 3.0, 4.0, 0.0, 31),
      (3.0, 4.0, 5.0, 1.0, 32),
      (4.0, 5.0, 6.0, 1.0, 33),
    ).toDF("feature1", "feature2", "feature3", "label", "base_margin")

    // Select the features and label columns
    val labelCol = "label"

    // Assemble the feature columns into a single vector column
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")
    val dataset = assembler.transform(df)

    val arrayInput = df.select(array(features.map(col(_)): _*).as("features"),
      col("label"), col("base_margin"))

    val est = new NewXGBoostClassifier()
      .setNumWorkers(1)
      .setLabelCol(labelCol)
      .setBaseMarginCol("base_margin")
    //    val est = new XGBoostClassifier().setLabelCol(labelCol)

    est.fit(arrayInput)

  }

}
