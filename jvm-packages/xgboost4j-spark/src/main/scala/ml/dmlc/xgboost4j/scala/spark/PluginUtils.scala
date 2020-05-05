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

import java.util.ServiceLoader
import scala.collection.JavaConverters._

private[spark] object PluginUtils {

  private def getClassLoader: ClassLoader =
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getClass.getClassLoader)

  def loadPlugin: TrainPlugin = {
    val serviceLoader = ServiceLoader.load(classOf[TrainPlugin], getClassLoader)
    val iter = serviceLoader.asScala

    if (iter.isEmpty) {
      // default to LabeledPointTrain
      new LabeledPointTrain
    } else {
      // only take first plugin
      iter.head
    }
  }

}
