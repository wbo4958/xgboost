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
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}

private[spark] class ModelImpl extends ModelPlugin {
  var useExternalMemory = false
  var appName = ""
  var missing = Float.NaN
  var allowNonZeroForMissing = false

  override def initialize(sc: SparkContext, model: Model[_], appName: String): Unit = {
    this.appName = appName
    model match {
      case classifierModel: XGBoostClassificationModel =>
        useExternalMemory = classifierModel.getUseExternalMemory
        allowNonZeroForMissing = classifierModel.getAllowNonZeroForMissingValue
      case _ => throw new IllegalArgumentException("not supported yet")
    }
  }

  override def extractRdd(model: Model[_], dataset: Dataset[_]): RDD[_] = {
    dataset.asInstanceOf[Dataset[Row]].rdd
  }

  override def buildDMatrix(batchCnt: Int, batches: Seq[_]): (DMatrix, Iterator[Row]) = {
    val batchRow: Seq[Row] = batches.asInstanceOf[Seq[Row]]
    val features = batchRow.iterator.map(row => row.getAs[Vector]("features"))

    import DataUtils._
    val cacheInfo = {
      if (useExternalMemory) {
        s"$appName-${TaskContext.get().stageId()}-dtest_cache-" +
          s"${TaskContext.getPartitionId()}-batch-$batchCnt"
      } else {
        null
      }
    }

    val dm = new DMatrix(
      XGBoost.processMissingValues(
        features.map(_.asXGB),
        missing,
        allowNonZeroForMissing
      ),
      cacheInfo)
    (dm, batchRow.toIterator)
  }
}
