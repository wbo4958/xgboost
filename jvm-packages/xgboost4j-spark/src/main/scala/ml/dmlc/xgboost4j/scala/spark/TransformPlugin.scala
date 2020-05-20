/*
 Copyright (c) 2014 by Contributors

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

import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.SparkContext
import org.apache.spark.ml.Model
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

trait TransformPlugin {

  def transformSchema(schema: StructType, logging: Boolean): StructType = {
    schema
  }

  def initialize(sc: SparkContext, model: Model[_], appName: String): Unit = {}

  def extractRdd(model: Model[_], dataset: Dataset[_]): RDD[_]

  def buildDMatrix(batchCnt: Int, batches: Seq[_]): (DMatrix, Iterator[Row])
}
