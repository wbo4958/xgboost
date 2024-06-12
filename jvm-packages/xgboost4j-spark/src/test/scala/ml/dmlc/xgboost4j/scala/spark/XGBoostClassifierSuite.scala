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

import java.io.{File, FileInputStream}

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}

import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.scalatest.funsuite.AnyFunSuite
import org.apache.commons.io.IOUtils

import org.apache.spark.Partitioner
import org.apache.spark.ml.feature.VectorAssembler
import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.parseJson

class XGBoostClassifierSuite extends AnyFunSuite with PerTest with TmpFolderPerSuite {

  test("") {

  }
}
