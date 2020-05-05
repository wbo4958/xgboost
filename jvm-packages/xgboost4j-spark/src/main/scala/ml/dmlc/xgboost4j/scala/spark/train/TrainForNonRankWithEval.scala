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
import ml.dmlc.xgboost4j.scala.spark.XGBoost.IteratorWrapper
import ml.dmlc.xgboost4j.scala.spark.{Watches}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

private[spark] case class TrainForNonRankWithEval(
    override protected val ifCacheDataBoolean: Boolean = false,
    missing: Float,
    allowNonZeroForMissing: Boolean,
    useExternalMemory: Boolean)
  extends LabeledPointImplIntf {

  override def rddTransform(
      trainSet: RDD[XGBLabeledPoint], numWorkers: Int,
      evalSets: Map[String, RDD[XGBLabeledPoint]]): RDD[_] = {
    val rdd = cacheData(trainSet).asInstanceOf[RDD[XGBLabeledPoint]]
    coPartitionNoGroupSets(rdd, evalSets, numWorkers)
  }

  override def buildWatches(iter: Iterator[_]): Watches = {
    val nameAndLabeledPointSets = iter
      .asInstanceOf[Iterator[(String, Iterator[XGBLabeledPoint])]]
    buildWatches(
      nameAndLabeledPointSets.map {
        case (name, iter) => (name, processMissingValues(iter, missing, allowNonZeroForMissing))
      },
      getCacheDirName(useExternalMemory))
  }

  private def buildWatches(
      nameAndLabeledPointSets: Iterator[(String, Iterator[XGBLabeledPoint])],
      cachedDirName: Option[String]): Watches = {
    val dms = nameAndLabeledPointSets.map {
      case (name, labeledPoints) =>
        val baseMargins = new mutable.ArrayBuilder.ofFloat
        val duplicatedItr = labeledPoints.map(labeledPoint => {
          baseMargins += labeledPoint.baseMargin
          labeledPoint
        })
        val dMatrix = new DMatrix(duplicatedItr, cachedDirName.map(_ + s"/$name").orNull)
        val baseMargin = fromBaseMarginsToArray(baseMargins.result().iterator)
        if (baseMargin.isDefined) {
          dMatrix.setBaseMargin(baseMargin.get)
        }
        (name, dMatrix)
    }.toArray
    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  private def coPartitionNoGroupSets(
      trainingData: RDD[XGBLabeledPoint],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int) = {
    // eval_sets is supposed to be set by the caller of [[trainDistributed]]
    val allDatasets = Map("train" -> trainingData) ++ evalSets
    val repartitionedDatasets = allDatasets.map { case (name, rdd) =>
      if (rdd.getNumPartitions != nWorkers) {
        (name, rdd.repartition(nWorkers))
      } else {
        (name, rdd)
      }
    }
    repartitionedDatasets.foldLeft(trainingData.sparkContext.parallelize(
      Array.fill[(String, Iterator[XGBLabeledPoint])](nWorkers)(null), nWorkers)) {
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter) {
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }
}
