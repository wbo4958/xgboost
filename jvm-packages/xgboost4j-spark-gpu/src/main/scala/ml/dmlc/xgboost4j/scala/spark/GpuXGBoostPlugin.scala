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

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.{asScalaIteratorConverter, seqAsJavaListConverter}

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.{ColumnarRdd, GpuColumnVectorUtils}
import org.apache.commons.logging.LogFactory
import org.apache.spark.ml.param.Param
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.TaskContext
import org.apache.spark.ml.functions.array_to_vector

import ml.dmlc.xgboost4j.java.{CudfColumnBatch, GpuColumnBatch}
import ml.dmlc.xgboost4j.scala.{DMatrix, QuantileDMatrix}
import ml.dmlc.xgboost4j.scala.spark.params.HasGroupCol

/**
 * GpuXGBoostPlugin is the XGBoost plugin which leverage spark-rapids
 * to accelerate the XGBoost from ETL to train.
 */
class GpuXGBoostPlugin extends XGBoostPlugin {

  private val logger = LogFactory.getLog("XGBoostSparkGpuPlugin")

  /**
   * Whether the plugin is enabled or not, if not enabled, fallback
   * to the regular CPU pipeline
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  override def isEnabled(dataset: Dataset[_]): Boolean = {
    val conf = dataset.sparkSession.conf
    val hasRapidsPlugin = conf.get("spark.sql.extensions", "").split(",").contains(
      "com.nvidia.spark.rapids.SQLExecPlugin")
    val rapidsEnabled = conf.get("spark.rapids.sql.enabled", "false").toBoolean
    hasRapidsPlugin && rapidsEnabled
  }

  // TODO, support numeric type
  private[spark] def preprocess[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M], dataset: Dataset[_]): Dataset[_] = {

    // Columns to be selected for XGBoost training
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    val schema = dataset.schema

    def selectCol(c: Param[String]) = {
      // TODO support numeric types
      if (estimator.isDefinedNonEmpty(c)) {
        selectedCols.append(estimator.castToFloatIfNeeded(schema, estimator.getOrDefault(c)))
      }
    }

    Seq(estimator.labelCol, estimator.weightCol, estimator.baseMarginCol).foreach(selectCol)
    estimator match {
      case p: HasGroupCol => selectCol(p.groupCol)
      case _ =>
    }

    // TODO support array/vector feature
    estimator.getFeaturesCols.foreach { name =>
      val col = estimator.castToFloatIfNeeded(dataset.schema, name)
      selectedCols.append(col)
    }
    val input = dataset.select(selectedCols: _*)
    estimator.repartitionIfNeeded(input)
  }

  // visiable for testing
  private[spark] def validate[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M],
      dataset: Dataset[_]): Unit = {
    require(estimator.getTreeMethod == "gpu_hist" || estimator.getDevice != "cpu",
      "Using Spark-Rapids to accelerate XGBoost must set device=cuda")
  }

  /**
   * Convert Dataset to RDD[Watches] which will be fed into XGBoost
   *
   * @param estimator which estimator to be handled.
   * @param dataset   to be converted.
   * @return RDD[Watches]
   */
  override def buildRddWatches[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M],
      dataset: Dataset[_]): RDD[Watches] = {

    validate(estimator, dataset)

    val train = preprocess(estimator, dataset)
    val schema = train.schema

    val indices = estimator.buildColumnIndices(schema)

    val maxBin = estimator.getMaxBins
    val nthread = estimator.getNthread
    val missing = estimator.getMissing

    /** build QuantilDMatrix on the executor side */
    def buildQuantileDMatrix(iter: Iterator[Table]): QuantileDMatrix = {
      val colBatchIter = iter.map { table =>
        withResource(new GpuColumnBatch(table, null)) { batch =>
          new CudfColumnBatch(
            batch.slice(indices.featureIds.get.map(Integer.valueOf).asJava),
            batch.slice(indices.labelId),
            batch.slice(indices.weightId.getOrElse(-1)),
            batch.slice(indices.marginId.getOrElse(-1)));
        }
      }
      new QuantileDMatrix(colBatchIter, missing, maxBin, nthread)
    }

    estimator.getEvalDataset().map { evalDs =>
      val evalProcessed = preprocess(estimator, evalDs)
      ColumnarRdd(train.toDF()).zipPartitions(ColumnarRdd(evalProcessed.toDF())) {
        (trainIter, evalIter) =>
          val trainDM = buildQuantileDMatrix(trainIter)
          val evalDM = buildQuantileDMatrix(evalIter)
          Iterator.single(new Watches(Array(trainDM, evalDM),
            Array(Utils.TRAIN_NAME, Utils.VALIDATION_NAME), None))
      }
    }.getOrElse(
      ColumnarRdd(train.toDF()).mapPartitions { iter =>
        val dm = buildQuantileDMatrix(iter)
        Iterator.single(new Watches(Array(dm), Array(Utils.TRAIN_NAME), None))
      }
    )
  }

  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } finally {
      r.close()
    }
  }


  override def transform[M <: XGBoostModel[M]](model: XGBoostModel[M],
                                               dataset: Dataset[_]): DataFrame = {
    val sc = dataset.sparkSession.sparkContext

    val (transformedSchema, pred) = model.preprocess(dataset)
    val bBooster = sc.broadcast(model.nativeBooster)
    val bOriginalSchema = sc.broadcast(dataset.schema)

    val featureIds = model.getFeaturesCols.distinct.map(dataset.schema.fieldIndex).toList
    val isLocal = sc.isLocal
    val missing = model.getMissing
    val nThread = model.getNthread

    val rdd = ColumnarRdd(dataset.asInstanceOf[DataFrame]).mapPartitions { tableIters =>
      // booster is visible for all spark tasks in the same executor
      val booster = bBooster.value
      val originalSchema = bOriginalSchema.value

      // UnsafeProjection is not serializable so do it on the executor side
      val toUnsafe = UnsafeProjection.create(originalSchema)

      synchronized {
        val device = booster.getAttr("device")
        if (device != null && device.trim.isEmpty) {
          booster.setAttr("device", "cuda")
          val gpuId = if (!isLocal) XGBoost.getGPUAddrFromResources else 0
          booster.setParam("device", s"cuda:$gpuId")
          logger.info("GPU transform on GPU device: " + gpuId)
        }
      }

      // Iterator on Row
      new Iterator[Row] {
        // Convert InternalRow to Row
        private val converter: InternalRow => Row = CatalystTypeConverters
          .createToScalaConverter(originalSchema)
          .asInstanceOf[InternalRow => Row]

        // GPU batches read in must be closed by the receiver
        @transient var currentBatch: ColumnarBatch = null

        // Iterator on Row
        var iter: Iterator[Row] = null

        TaskContext.get().addTaskCompletionListener[Unit](_ => {
          closeCurrentBatch() // close the last ColumnarBatch
        })

        private def closeCurrentBatch(): Unit = {
          if (currentBatch != null) {
            currentBatch.close()
            currentBatch = null
          }
        }

        def loadNextBatch(): Unit = {
          closeCurrentBatch()
          if (tableIters.hasNext) {
            val dataTypes = originalSchema.fields.map(x => x.dataType)
            iter = withResource(tableIters.next()) { table =>
              val gpuColumnBatch = new GpuColumnBatch(table, originalSchema)
              // Create DMatrix
              val featureTable = gpuColumnBatch.slice(featureIds.map(Integer.valueOf).asJava)
              if (featureTable == null) {
                throw new RuntimeException("Something wrong for feature indices")
              }
              try {
                val cudfColumnBatch = new CudfColumnBatch(featureTable, null, null, null)
                val dm = new DMatrix(cudfColumnBatch, missing, nThread)
                if (dm == null) {
                  Iterator.empty
                } else {
                  try {
                    currentBatch = new ColumnarBatch(
                      GpuColumnVectorUtils.extractHostColumns(table, dataTypes),
                      table.getRowCount().toInt)
                    val rowIterator = currentBatch.rowIterator().asScala.map(toUnsafe)
                      .map(converter(_))
                    model.predictInternal(booster, dm, pred, rowIterator).toIterator
                  } finally {
                    dm.delete()
                  }
                }
              } finally {
                featureTable.close()
              }
            }
          } else {
            iter = null
          }
        }

        override def hasNext: Boolean = {
          val itHasNext = iter != null && iter.hasNext
          if (!itHasNext) { // Don't have extra Row for current ColumnarBatch
            loadNextBatch()
            iter != null && iter.hasNext
          } else {
            itHasNext
          }
        }

        override def next(): Row = {
          if (iter == null || !iter.hasNext) {
            loadNextBatch()
          }
          if (iter == null) {
            throw new NoSuchElementException()
          }
          iter.next()
        }
      }
    }
    bBooster.unpersist(false)
    bOriginalSchema.unpersist(false)
    var output = dataset.sparkSession.createDataFrame(rdd, transformedSchema)

    // Convert leaf/contrib to the vector from array
    if (pred.predLeaf) {
      output = output.withColumn(model.getLeafPredictionCol,
        array_to_vector(output.col(model.getLeafPredictionCol)))
    }

    if (pred.predContrib) {
      output = output.withColumn(model.getContribPredictionCol,
        array_to_vector(output.col(model.getContribPredictionCol)))
    }

    model.postTransform(output).toDF()
  }
}
