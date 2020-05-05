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

package ml.dmlc.xgboost4j.scala.spark.train

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.{Watches}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.util.Random

private[spark] case class TrainForNonRankWithoutEval(
    override protected val ifCacheDataBoolean: Boolean = false,
    missing: Float,
    allowNonZeroForMissing: Boolean,
    useExternalMemory: Boolean,
    trainTestRatio: Double,
    seed: Long) extends LabeledPointImplIntf {

  override def rddTransform(
      trainSet: RDD[XGBLabeledPoint], numWorkers: Int,
      evalSets: Map[String, RDD[XGBLabeledPoint]]): RDD[_] = {
    cacheData(trainSet).asInstanceOf[RDD[XGBLabeledPoint]]
  }

  override def buildWatches(iter: Iterator[_]): Watches = {
    val labeledPoints = iter.asInstanceOf[Iterator[XGBLabeledPoint]]
    buildWatches(trainTestRatio, seed,
      processMissingValues(labeledPoints, missing, allowNonZeroForMissing),
      getCacheDirName(useExternalMemory))
  }

  private def buildWatches(
      trainTestRatio: Double,
      seed: Long,
      labeledPoints: Iterator[XGBLabeledPoint],
      cacheDirName: Option[String]): Watches = {
    val r = new Random(seed)
    val testPoints = mutable.ArrayBuffer.empty[XGBLabeledPoint]
    val trainBaseMargins = new mutable.ArrayBuilder.ofFloat
    val testBaseMargins = new mutable.ArrayBuilder.ofFloat
    val trainPoints = labeledPoints.filter { labeledPoint =>
      val accepted = r.nextDouble() <= trainTestRatio
      if (!accepted) {
        testPoints += labeledPoint
        testBaseMargins += labeledPoint.baseMargin
      } else {
        trainBaseMargins += labeledPoint.baseMargin
      }
      accepted
    }
    val trainMatrix = new DMatrix(trainPoints, cacheDirName.map(_ + "/train").orNull)
    val testMatrix = new DMatrix(testPoints.iterator, cacheDirName.map(_ + "/test").orNull)

    val trainMargin = fromBaseMarginsToArray(trainBaseMargins.result().iterator)
    val testMargin = fromBaseMarginsToArray(testBaseMargins.result().iterator)
    if (trainMargin.isDefined) trainMatrix.setBaseMargin(trainMargin.get)
    if (testMargin.isDefined) testMatrix.setBaseMargin(testMargin.get)

    new Watches(Array(trainMatrix, testMatrix), Array("train", "test"), cacheDirName)
  }
}
