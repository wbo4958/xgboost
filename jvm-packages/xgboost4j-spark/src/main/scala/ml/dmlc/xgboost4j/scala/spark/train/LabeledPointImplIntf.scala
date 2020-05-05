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

import java.nio.file.Files

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.scala.spark.{LabeledPointGroupIterator, Watches, XGBLabeledPointGroup}
import org.apache.commons.logging.LogFactory
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

private[spark] trait LabeledPointImplIntf {
  protected val logger = LogFactory.getLog("LabeledPointImpl")

  protected val ifCacheDataBoolean: Boolean

  protected var cachedRdd: RDD[_] = _

  protected def cacheData(input: RDD[_]): RDD[_] = {
    if (ifCacheDataBoolean) input.persist(StorageLevel.MEMORY_AND_DISK) else input
  }

  def cleanUp: Unit = {
    if (ifCacheDataBoolean && cachedRdd != null) cachedRdd.unpersist()
  }

  protected def fromBaseMarginsToArray(baseMargins: Iterator[Float]): Option[Array[Float]] = {
    val builder = new mutable.ArrayBuilder.ofFloat()
    var nTotal = 0
    var nUndefined = 0
    while (baseMargins.hasNext) {
      nTotal += 1
      val baseMargin = baseMargins.next()
      if (baseMargin.isNaN) {
        nUndefined += 1  // don't waste space for all-NaNs.
      } else {
        builder += baseMargin
      }
    }
    if (nUndefined == nTotal) {
      None
    } else if (nUndefined == 0) {
      Some(builder.result())
    } else {
      throw new IllegalArgumentException(
        s"Encountered a partition with $nUndefined NaN base margin values. " +
          s"If you want to specify base margin, ensure all values are non-NaN.")
    }
  }

  protected def getCacheDirName(useExternalMemory: Boolean): Option[String] = {
    val taskId = TaskContext.getPartitionId().toString
    if (useExternalMemory) {
      val dir = Files.createTempDirectory(s"${TaskContext.get().stageId()}-cache-$taskId")
      Some(dir.toAbsolutePath.toString)
    } else {
      None
    }
  }

  private def verifyMissingSetting(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      allowNonZeroMissing: Boolean): Iterator[XGBLabeledPoint] = {
    if (missing != 0.0f && !allowNonZeroMissing) {
      xgbLabelPoints.map(labeledPoint => {
        if (labeledPoint.indices != null) {
          throw new RuntimeException(s"you can only specify missing value as 0.0 (the currently" +
            s" set value $missing) when you have SparseVector or Empty vector as your feature" +
            s" format. If you didn't use Spark's VectorAssembler class to build your feature " +
            s"vector but instead did so in a way that preserves zeros in your feature vector " +
            s"you can avoid this check by using the 'allow_non_zero_missing parameter'" +
            s" (only use if you know what you are doing)")
        }
        labeledPoint
      })
    } else {
      xgbLabelPoints
    }
  }

  private def removeMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      keepCondition: Float => Boolean): Iterator[XGBLabeledPoint] = {
    xgbLabelPoints.map { labeledPoint =>
      val indicesBuilder = new mutable.ArrayBuilder.ofInt()
      val valuesBuilder = new mutable.ArrayBuilder.ofFloat()
      for ((value, i) <- labeledPoint.values.zipWithIndex if keepCondition(value)) {
        indicesBuilder += (if (labeledPoint.indices == null) i else labeledPoint.indices(i))
        valuesBuilder += value
      }
      labeledPoint.copy(indices = indicesBuilder.result(), values = valuesBuilder.result())
    }
  }

  private[spark] def processMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      allowNonZeroMissing: Boolean): Iterator[XGBLabeledPoint] = {
    if (!missing.isNaN) {
      removeMissingValues(verifyMissingSetting(xgbLabelPoints, missing, allowNonZeroMissing),
        missing, (v: Float) => v != missing)
    } else {
      removeMissingValues(verifyMissingSetting(xgbLabelPoints, missing, allowNonZeroMissing),
        missing, (v: Float) => !v.isNaN)
    }
  }

  protected def processMissingValuesWithGroup(
      xgbLabelPointGroups: Iterator[Array[XGBLabeledPoint]],
      missing: Float,
      allowNonZeroMissing: Boolean): Iterator[Array[XGBLabeledPoint]] = {
    if (!missing.isNaN) {
      xgbLabelPointGroups.map {
        labeledPoints => processMissingValues(
          labeledPoints.iterator,
          missing,
          allowNonZeroMissing
        ).toArray
      }
    } else {
      xgbLabelPointGroups
    }
  }

  protected def repartitionForTrainingGroup(
      trainingData: RDD[XGBLabeledPoint], nWorkers: Int): RDD[Array[XGBLabeledPoint]] = {
    val allGroups = aggByGroupInfo(trainingData)
    logger.info(s"repartitioning training group set to $nWorkers partitions")
    allGroups.repartition(nWorkers)
  }

  protected def aggByGroupInfo(trainingData: RDD[XGBLabeledPoint]) = {
    val normalGroups: RDD[Array[XGBLabeledPoint]] = trainingData.mapPartitions(
      // LabeledPointGroupIterator returns (Boolean, Array[XGBLabeledPoint])
      new LabeledPointGroupIterator(_)).filter(!_.isEdgeGroup).map(_.points)

    // edge groups with partition id.
    val edgeGroups: RDD[(Int, XGBLabeledPointGroup)] = trainingData.mapPartitions(
      new LabeledPointGroupIterator(_)).filter(_.isEdgeGroup).map(
      group => (TaskContext.getPartitionId(), group))

    // group chunks from different partitions together by group id in XGBLabeledPoint.
    // use groupBy instead of aggregateBy since all groups within a partition have unique group ids.
    val stitchedGroups: RDD[Array[XGBLabeledPoint]] = edgeGroups.groupBy(_._2.groupId).map(
      groups => {
        val it: Iterable[(Int, XGBLabeledPointGroup)] = groups._2
        // sorted by partition id and merge list of Array[XGBLabeledPoint] into one array
        it.toArray.sortBy(_._1).flatMap(_._2.points)
      })
    normalGroups.union(stitchedGroups)
  }

  def rddTransform(trainSet: RDD[XGBLabeledPoint], numWorkers: Int,
      evalSets: Map[String, RDD[XGBLabeledPoint]] = Map.empty): RDD[_]

  def buildWatches(iter: Iterator[_]): Watches
}
