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

package org.apache.spark.ml.xgboost

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DatasetUtils, DefaultParamsReader, DefaultParamsWriter, SchemaUtils}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import org.json4s.{JObject, JValue}

object SparkUtils {

  def getNumClasses(dataset: Dataset[_], labelCol: String, maxNumClasses: Int = 100): Int = {
    DatasetUtils.getNumClasses(dataset, labelCol, maxNumClasses)
  }

  def checkNumericType(schema: StructType, colName: String, msg: String = ""): Unit = {
    SchemaUtils.checkNumericType(schema, colName, msg)
  }

  def saveMetadata(
                    instance: Params,
                    path: String,
                    sc: SparkContext,
                    extraMetadata: Option[JObject] = None,
                    paramMap: Option[JValue] = None): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc, extraMetadata, paramMap)
  }

  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    DefaultParamsReader.loadMetadata(path, sc, expectedClassName)
  }


}
