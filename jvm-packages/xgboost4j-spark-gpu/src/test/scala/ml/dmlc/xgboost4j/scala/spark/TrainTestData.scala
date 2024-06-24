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

import scala.io.Source

trait TrainTestData {
  protected def getResourceLines(resource: String): Iterator[String] = {
    require(resource.startsWith("/"), "resource must start with /")
    val is = getClass.getResourceAsStream(resource)
    if (is == null) {
      sys.error(s"failed to resolve resource $resource")
    }
    Source.fromInputStream(is).getLines()
  }
}

object MultiClassification extends TrainTestData {
  val
}
