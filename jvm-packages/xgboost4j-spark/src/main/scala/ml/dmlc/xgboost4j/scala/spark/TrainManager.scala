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

import java.sql.Struct

import ml.dmlc.xgboost4j.scala.{Booster, EvalTrait, ExternalCheckpointManager, ExternalCheckpointParams, ObjectiveTrait, spark, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker
import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, XGBoostError, RabitTracker => PyRabitTracker}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.ml.Estimator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.{SparkContext, SparkParallelismTracker, TaskContext}
import org.apache.spark.sql.{Dataset, SparkSession}

private[spark] object TrainManager {
  private val logger = LogFactory.getLog("XGBoostManager")

  val trainImpl: TrainPlugin = PluginUtils.loadPlugin

  /**
   * Check to see if Spark expects SSL encryption (`spark.ssl.enabled` set to true).
   * If so, throw an exception unless this safety measure has been explicitly overridden
   * via conf `xgboost.spark.ignoreSsl`.
   */
  private def validateSparkSslConf(sc: SparkContext): Unit = {
    val (sparkSslEnabled: Boolean, xgboostSparkIgnoreSsl: Boolean) =
      SparkSession.getActiveSession match {
        case Some(ss) =>
          (ss.conf.getOption("spark.ssl.enabled").getOrElse("false").toBoolean,
            ss.conf.getOption("xgboost.spark.ignoreSsl").getOrElse("false").toBoolean)
        case None =>
          (sc.getConf.getBoolean("spark.ssl.enabled", false),
            sc.getConf.getBoolean("xgboost.spark.ignoreSsl", false))
      }
    if (sparkSslEnabled) {
      if (xgboostSparkIgnoreSsl) {
        logger.warn(s"spark-xgboost is being run without encrypting data in transit!  " +
          s"Spark Conf spark.ssl.enabled=true was overridden with xgboost.spark.ignoreSsl=true.")
      } else {
        throw new Exception("xgboost-spark found spark.ssl.enabled=true to encrypt data " +
          "in transit, but xgboost-spark sends non-encrypted data over the wire for efficiency. " +
          "To override this protection and still use xgboost-spark at your own risk, " +
          "you can set the SparkSession conf to use xgboost.spark.ignoreSsl=true.")
      }
    }
  }

  private case class XGBoostParams(
      numWorkers: Int,
      numRounds: Int,
      useExternalMemory: Boolean,
      obj: ObjectiveTrait,
      eval: EvalTrait,
      missing: Float,
      allowNonZeroForMissing: Boolean,
      trackerConf: TrackerConf,
      timeoutRequestWorkers: Long,
      checkpointParam: Option[ExternalCheckpointParams],
      xgbInputParams: XGBoostExecutionInputParams,
      earlyStoppingParams: XGBoostExecutionEarlyStoppingParams,
      cacheTrainingSet: Boolean)

  private def parseParams(rawParams: Map[String, Any], sc: SparkContext): XGBoostParams = {
    validateSparkSslConf(sc)
    val nWorkers = rawParams("num_workers").asInstanceOf[Int]
    require(nWorkers > 0, "you must specify more than 0 workers")

    val round = rawParams("num_round").asInstanceOf[Int]
    val useExternalMemory = rawParams("use_external_memory").asInstanceOf[Boolean]

    val obj = rawParams.getOrElse("custom_obj", null).asInstanceOf[ObjectiveTrait]
    if (obj != null) {
      require(rawParams.get("objective_type").isDefined, "parameter \"objective_type\" " +
        "is not defined, you have to specify the objective type as classification or regression" +
        " with a customized objective function")
    }

    val eval = rawParams.getOrElse("custom_eval", null).asInstanceOf[EvalTrait]
    val missing = rawParams.getOrElse("missing", Float.NaN).asInstanceOf[Float]
    val allowNonZeroForMissing = rawParams
      .getOrElse("allow_non_zero_for_missing", false)
      .asInstanceOf[Boolean]

    if (rawParams.contains("train_test_ratio")) {
      logger.warn("train_test_ratio is deprecated since XGBoost 0.82, we recommend to explicitly" +
        " pass a training and multiple evaluation datasets by passing 'eval_sets' and " +
        "'eval_set_names'")
    }

    val trackerConf = rawParams.get("tracker_conf") match {
      case None => TrackerConf()
      case Some(conf: TrackerConf) => conf
      case _ => throw new IllegalArgumentException("parameter \"tracker_conf\" must be an " +
        "instance of TrackerConf.")
    }
    val timeoutRequestWorkers: Long = rawParams.get("timeout_request_workers") match {
      case None => 0L
      case Some(interval: Long) => interval
      case _ => throw new IllegalArgumentException("parameter \"timeout_request_workers\" must be" +
        " an instance of Long.")
    }
    val checkpointParam =
      ExternalCheckpointParams.extractParams(rawParams)

    val trainTestRatio = rawParams.getOrElse("train_test_ratio", 1.0)
      .asInstanceOf[Double]
    val seed = rawParams.getOrElse("seed", System.nanoTime()).asInstanceOf[Long]
    val inputParams = XGBoostExecutionInputParams(trainTestRatio, seed)

    val earlyStoppingRounds = rawParams.getOrElse(
      "num_early_stopping_rounds", 0).asInstanceOf[Int]
    val maximizeEvalMetrics = rawParams.getOrElse(
      "maximize_evaluation_metrics", true).asInstanceOf[Boolean]
    val xgbExecEarlyStoppingParams = XGBoostExecutionEarlyStoppingParams(earlyStoppingRounds,
      maximizeEvalMetrics)

    val cacheTrainingSet = rawParams.getOrElse("cache_training_set", false)
      .asInstanceOf[Boolean]

    XGBoostParams(nWorkers, round, useExternalMemory, obj, eval, missing, allowNonZeroForMissing,
      trackerConf, timeoutRequestWorkers, checkpointParam, inputParams, xgbExecEarlyStoppingParams,
      cacheTrainingSet)
  }

  private[spark] def transform(clsifier: XGBoostClassifier, schema: StructType,
      logging: Boolean = false): StructType = {
    trainImpl.transform(clsifier, schema)
  }

  /**
   * @return A tuple of the booster and the metrics used to build training summary
   */
  @throws(classOf[XGBoostError])
  private[spark] def trainDistributed(
      est: Estimator[_],
      dataset: Dataset[_],
      params: Map[String, Any],
      rawParams: Map[String, Any],
      hasGroup: Boolean = false,
      evalSetsMap: Map[String, Dataset[_]] = Map.empty): (Booster, Map[String, Array[Float]]) = {
    logger.info(s"Running XGBoost ${spark.VERSION} with parameters:\n${params.mkString("\n")}")

    val sc = dataset.sparkSession.sparkContext
    val xgbExecParams = parseParams(params, sc)

    // todo add transformSchema
    trainImpl.initialize(params, rawParams, hasGroup, evalSetsMap.nonEmpty, sc)

    val prevBooster = xgbExecParams.checkpointParam.map { checkpointParam =>
      val checkpointManager = new ExternalCheckpointManager(
        checkpointParam.checkpointPath,
        FileSystem.get(sc.hadoopConfiguration))
      checkpointManager.cleanUpHigherVersions(xgbExecParams.numRounds)
      checkpointManager.loadCheckpointAsScalaBooster()
    }.orNull

    try {
      // Train for every ${savingRound} rounds and save the partially completed booster
      val tracker = startTracker(xgbExecParams.numWorkers, xgbExecParams.trackerConf)

      val (booster, metrics) = try {
        val parallelismTracker = new SparkParallelismTracker(sc,
          xgbExecParams.timeoutRequestWorkers,
          xgbExecParams.numWorkers)

        val rabitEnv = tracker.getWorkerEnvs

        // we don't care about if it's ranking or non-ranking
        // what we care about is how to convert dataset and how to build Watches

        val boostersAndMetrics = trainImpl.convertDatasetToRdd(est, params, dataset, evalSetsMap)
          .mapPartitions(iter => {

            val watches = trainImpl.buildWatches(iter)

            // to workaround the empty partitions in training dataset,
            // this might not be the best efficient implementation, see
            // (https://github.com/dmlc/xgboost/issues/1277)
            if (watches.toMap("train").rowNum == 0) {
              throw new XGBoostError(
                s"detected an empty partition in the training data, partition ID:" +
                  s" ${TaskContext.getPartitionId()}")
            }

            // prepare rabit env
            val taskId = TaskContext.getPartitionId().toString
            val attempt = TaskContext.get().attemptNumber.toString
            rabitEnv.put("DMLC_TASK_ID", taskId)
            rabitEnv.put("DMLC_NUM_ATTEMPT", attempt)
            rabitEnv.put("DMLC_WORKER_STOP_PROCESS_ON_ERROR", "false")

            val numRounds = xgbExecParams.numRounds
            val makeCheckpoint = xgbExecParams.checkpointParam.isDefined && taskId.toInt == 0
            try {
              trainImpl.onPreRabitInit
              Rabit.init(rabitEnv)
              trainImpl.onPostRabitInit

              val numEarlyStoppingRounds = xgbExecParams.earlyStoppingParams.numEarlyStoppingRounds
              val metrics = Array.tabulate(watches.size)(_ => Array.ofDim[Float](numRounds))
              val externalCheckpointParams = xgbExecParams.checkpointParam

              trainImpl.onPreXGboostTrain
              val finalParams = trainImpl.getFinalXGBoostParam(params)
              val booster = if (makeCheckpoint) {
                SXGBoost.trainAndSaveCheckpoint(
                  watches.toMap("train"), finalParams, numRounds,
                  watches.toMap, metrics, xgbExecParams.obj, xgbExecParams.eval,
                  earlyStoppingRound = numEarlyStoppingRounds, prevBooster,
                  externalCheckpointParams)
              } else {
                SXGBoost.train(watches.toMap("train"), finalParams, numRounds,
                  watches.toMap, metrics, xgbExecParams.obj, xgbExecParams.eval,
                  earlyStoppingRound = numEarlyStoppingRounds, prevBooster)
              }

              trainImpl.onPostXGBoostTrain
              Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
            } catch {
              case xgbException: XGBoostError =>
                logger.error(s"XGBooster worker $taskId has failed $attempt times due to ",
                  xgbException)
                trainImpl.onExecutorError(xgbException)
                throw xgbException
            } finally {
              trainImpl.onExecutorCleanUp
              Rabit.shutdown()
              watches.delete()
            }

          }).cache()

        val sparkJobThread = new Thread() {
          override def run() {
            // force the job
            boostersAndMetrics.foreachPartition(() => _)
          }
        }
        sparkJobThread.setUncaughtExceptionHandler(tracker)
        sparkJobThread.start()
        val trackerReturnVal = parallelismTracker.execute(tracker.waitFor(0L))
        logger.info(s"Rabit returns with exit code $trackerReturnVal")
        val (booster, metrics) = postTrackerReturnProcessing(trackerReturnVal,
          boostersAndMetrics, sparkJobThread)
        (booster, metrics)
      } finally {
        tracker.stop()
      }
      // we should delete the checkpoint directory after a successful training
      xgbExecParams.checkpointParam.foreach {
        cpParam =>
          if (!xgbExecParams.checkpointParam.get.skipCleanCheckpoint) {
            val checkpointManager = new ExternalCheckpointManager(
              cpParam.checkpointPath,
              FileSystem.get(sc.hadoopConfiguration))
            checkpointManager.cleanPath()
          }
      }
      (booster, metrics)
    } catch {
      case t: Throwable =>
        // if the job was aborted due to an exception
        logger.error("the job was aborted due to ", t)
        sc.stop()
        throw t
    } finally {
      trainImpl.onDriverCleanUp
    }
  }

  private def startTracker(nWorkers: Int, trackerConf: TrackerConf): IRabitTracker = {
    val tracker: IRabitTracker = trackerConf.trackerImpl match {
      case "scala" => new RabitTracker(nWorkers)
      case "python" => new PyRabitTracker(nWorkers)
      case _ => new PyRabitTracker(nWorkers)
    }

    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

  private def postTrackerReturnProcessing(
      trackerReturnVal: Int,
      distributedBoostersAndMetrics: RDD[(Booster, Map[String, Array[Float]])],
      sparkJobThread: Thread): (Booster, Map[String, Array[Float]]) = {
    if (trackerReturnVal == 0) {
      // Copies of the final booster and the corresponding metrics
      // reside in each partition of the `distributedBoostersAndMetrics`.
      // Any of them can be used to create the model.
      // it's safe to block here forever, as the tracker has returned successfully, and the Spark
      // job should have finished, there is no reason for the thread cannot return
      sparkJobThread.join()
      val (booster, metrics) = distributedBoostersAndMetrics.first()
      distributedBoostersAndMetrics.unpersist(false)
      (booster, metrics)
    } else {
      try {
        if (sparkJobThread.isAlive) {
          sparkJobThread.interrupt()
        }
      } catch {
        case _: InterruptedException =>
          logger.info("spark job thread is interrupted")
      }
      throw new XGBoostError("XGBoostModel training failed")
    }
  }
}
