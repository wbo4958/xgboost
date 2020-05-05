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
import ml.dmlc.xgboost4j.scala.spark.XGBoost.{IteratorWrapper}
import ml.dmlc.xgboost4j.scala.spark.{Watches}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

private[spark] case class TrainForRankWithEval(
    override protected val ifCacheDataBoolean: Boolean = false,
    missing: Float,
    allowNonZeroForMissing: Boolean,
    useExternalMemory: Boolean) extends LabeledPointImplIntf {

  override def rddTransform(
      trainSet: RDD[XGBLabeledPoint], numWorkers: Int,
      evalSets: Map[String, RDD[XGBLabeledPoint]]): RDD[_] = {
    val repartitionedData = repartitionForTrainingGroup(trainSet, numWorkers)
    val trainRdd = cacheData(repartitionedData)
      .asInstanceOf[RDD[Array[XGBLabeledPoint]]]
    coPartitionGroupSets(trainRdd, evalSets, numWorkers)
  }

  override def buildWatches(iter: Iterator[_]): Watches = {
    val labeledPointGroupSets = iter
      .asInstanceOf[Iterator[(String, Iterator[Array[XGBLabeledPoint]])]]
    buildWatchesWithGroup(
      labeledPointGroupSets.map {
        case (name, iter) => (name, processMissingValuesWithGroup(iter,
          missing, allowNonZeroForMissing))
      },
      getCacheDirName(useExternalMemory))
  }

  private def buildWatchesWithGroup(
      nameAndlabeledPointGroupSets: Iterator[(String, Iterator[Array[XGBLabeledPoint]])],
      cachedDirName: Option[String]): Watches = {
    val dms = nameAndlabeledPointGroupSets.map {
      case (name, labeledPointsGroups) =>
        val baseMargins = new mutable.ArrayBuilder.ofFloat
        val groupsInfo = new mutable.ArrayBuilder.ofInt
        val weights = new mutable.ArrayBuilder.ofFloat
        val iter = labeledPointsGroups.filter(labeledPointGroup => {
          var groupWeight = -1.0f
          var groupSize = 0
          labeledPointGroup.map { labeledPoint => {
            if (groupWeight < 0) {
              groupWeight = labeledPoint.weight
            } else if (groupWeight != labeledPoint.weight) {
              throw new IllegalArgumentException("the instances in the same group have to be" +
                s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
            }
            baseMargins += labeledPoint.baseMargin
            groupSize += 1
            labeledPoint
          }
          }
          weights += groupWeight
          groupsInfo += groupSize
          true
        })
        val dMatrix = new DMatrix(iter.flatMap(_.iterator), cachedDirName.map(_ + s"/$name").orNull)
        val baseMargin = fromBaseMarginsToArray(baseMargins.result().iterator)
        if (baseMargin.isDefined) {
          dMatrix.setBaseMargin(baseMargin.get)
        }
        dMatrix.setGroup(groupsInfo.result())
        dMatrix.setWeight(weights.result())
        (name, dMatrix)
    }.toArray
    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  private def coPartitionGroupSets(
      aggedTrainingSet: RDD[Array[XGBLabeledPoint]],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int): RDD[(String, Iterator[Array[XGBLabeledPoint]])] = {
    val repartitionedDatasets = Map("train" -> aggedTrainingSet) ++ evalSets.map {
      case (name, rdd) => {
        val aggedRdd = aggByGroupInfo(rdd)
        if (aggedRdd.getNumPartitions != nWorkers) {
          name -> aggedRdd.repartition(nWorkers)
        } else {
          name -> aggedRdd
        }
      }
    }
    repartitionedDatasets.foldLeft(aggedTrainingSet.sparkContext.parallelize(
      Array.fill[(String, Iterator[Array[XGBLabeledPoint]])](nWorkers)(null), nWorkers)) {
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
