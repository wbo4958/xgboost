package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.java.{Communicator, ITracker, RabitTracker, XGBoostError}
import ml.dmlc.xgboost4j.scala.spark.XGBoost.getGPUAddrFromResources
import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, Watches}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import org.apache.commons.logging.LogFactory
import org.apache.spark.rdd.RDD
import org.apache.spark.resource.{ResourceProfileBuilder, TaskResourceRequests}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext, TaskContext}


private[scala] case class RuntimeParams(
                                         numWorkers: Int,
                                         numRounds: Int,
                                         obj: ObjectiveTrait,
                                         eval: EvalTrait,
                                         trackerConf: TrackerConf,
                                         earlyStoppingRounds: Int,
                                         device: Option[String],
                                         isLocal: Boolean,
                                         featureNames: Option[Array[String]],
                                         featureTypes: Option[Array[String]],
                                         runOnGpu: Boolean) {
  private var rawParamMap: Map[String, Any] = _

  def setRawParamMap(inputMap: Map[String, Any]) {
    rawParamMap = inputMap
  }

  def toMap: Map[String, Any] = rawParamMap
}

private[this] class ParamsFactory(rawParams: Map[String, Any], sc: SparkContext) {
  private val logger = LogFactory.getLog("XGBoostSpark")
  private val isLocal = sc.isLocal
  private val overridedParams = overrideParams(rawParams, sc)

  validateSparkSslConf()

  /**
   * Check to see if Spark expects SSL encryption (`spark.ssl.enabled` set to true).
   * If so, throw an exception unless this safety measure has been explicitly overridden
   * via conf `xgboost.spark.ignoreSsl`.
   */
  private def validateSparkSslConf(): Unit = {
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

  /**
   * we should not include any nested structure in the output of this function as the map is
   * eventually to be feed to xgboost4j layer
   */
  private def overrideParams(params: Map[String, Any],
                             sc: SparkContext): Map[String, Any] = {
    val coresPerTask = sc.getConf.getInt("spark.task.cpus", 1)
    var overridedParams = params
    if (overridedParams.contains("nthread")) {
      val nThread = overridedParams("nthread").toString.toInt
      require(nThread <= coresPerTask,
        s"the nthread configuration ($nThread) must be smaller than or equal to " +
          s"spark.task.cpus ($coresPerTask)")
    } else {
      overridedParams = overridedParams + ("nthread" -> coresPerTask)
    }

    val numEarlyStoppingRounds = overridedParams.getOrElse(
      "num_early_stopping_rounds", 0).asInstanceOf[Int]
    overridedParams += "num_early_stopping_rounds" -> numEarlyStoppingRounds
    if (numEarlyStoppingRounds > 0 && overridedParams.getOrElse("custom_eval", null) != null) {
      throw new IllegalArgumentException("custom_eval does not support early stopping")
    }
    overridedParams
  }

  /**
   * The Map parameters accepted by estimator's constructor may have string type,
   * Eg, Map("num_workers" -> "6", "num_round" -> 5), we need to convert these
   * kind of parameters into the correct type in the function.
   *
   * @return RuntimeParams
   */
  def runtimeParams: RuntimeParams = {
    val obj = overridedParams.getOrElse("custom_obj", null).asInstanceOf[ObjectiveTrait]
    val eval = overridedParams.getOrElse("custom_eval", null).asInstanceOf[EvalTrait]
    if (obj != null) {
      require(overridedParams.get("objective_type").isDefined, "parameter \"objective_type\" " +
        "is not defined, you have to specify the objective type as classification or regression" +
        " with a customized objective function")
    }

    val nWorkers = overridedParams("num_workers").asInstanceOf[Int]
    val round = overridedParams("num_round").asInstanceOf[Int]

    val treeMethod: Option[String] = overridedParams.get("tree_method").map(_.toString)
    val device: Option[String] = overridedParams.get("device").map(_.toString)
    val deviceIsGpu = device.exists(_ == "cuda")

    require(!(treeMethod.exists(_ == "approx") && deviceIsGpu),
      "The tree method \"approx\" is not yet supported for Spark GPU cluster")

    // back-compatible with "gpu_hist"
    val runOnGpu = treeMethod.exists(_ == "gpu_hist") || deviceIsGpu

    val trackerConf = overridedParams.get("tracker_conf") match {
      case None => TrackerConf()
      case Some(conf: TrackerConf) => conf
      case _ => throw new IllegalArgumentException("parameter \"tracker_conf\" must be an " +
        "instance of TrackerConf.")
    }

    val earlyStoppingRounds = overridedParams.getOrElse(
      "num_early_stopping_rounds", 0).asInstanceOf[Int]

    val featureNames = if (overridedParams.contains("feature_names")) {
      Some(overridedParams("feature_names").asInstanceOf[Array[String]])
    } else None
    val featureTypes = if (overridedParams.contains("feature_types")) {
      Some(overridedParams("feature_types").asInstanceOf[Array[String]])
    } else None

    val xgbExecParam = RuntimeParams(
      nWorkers,
      round,
      obj,
      eval,
      trackerConf,
      earlyStoppingRounds,
      device,
      isLocal,
      featureNames,
      featureTypes,
      runOnGpu
    )
    xgbExecParam.setRawParamMap(overridedParams)
    xgbExecParam
  }
}

/**
 * A trait to manage stage-level scheduling
 */
private[spark] trait StageLevelScheduling extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private[spark] def isStandaloneOrLocalCluster(conf: SparkConf): Boolean = {
    val master = conf.get("spark.master")
    master != null && (master.startsWith("spark://") || master.startsWith("local-cluster"))
  }

  /**
   * To determine if stage-level scheduling should be skipped according to the spark version
   * and spark configurations
   *
   * @param sparkVersion spark version
   * @param runOnGpu     if xgboost training run on GPUs
   * @param conf         spark configurations
   * @return Boolean to skip stage-level scheduling or not
   */
  private[spark] def skipStageLevelScheduling(
                                               sparkVersion: String,
                                               runOnGpu: Boolean,
                                               conf: SparkConf): Boolean = {
    if (runOnGpu) {
      if (sparkVersion < "3.4.0") {
        logger.info("Stage-level scheduling in xgboost requires spark version 3.4.0+")
        return true
      }

      if (!isStandaloneOrLocalCluster(conf)) {
        logger.info("Stage-level scheduling in xgboost requires spark standalone or " +
          "local-cluster mode")
        return true
      }

      val executorCores = conf.getInt("spark.executor.cores", -1)
      val executorGpus = conf.getInt("spark.executor.resource.gpu.amount", -1)
      if (executorCores == -1 || executorGpus == -1) {
        logger.info("Stage-level scheduling in xgboost requires spark.executor.cores, " +
          "spark.executor.resource.gpu.amount to be set.")
        return true
      }

      if (executorCores == 1) {
        logger.info("Stage-level scheduling in xgboost requires spark.executor.cores > 1")
        return true
      }

      if (executorGpus > 1) {
        logger.info("Stage-level scheduling in xgboost will not work " +
          "when spark.executor.resource.gpu.amount > 1")
        return true
      }

      val taskGpuAmount = conf.getDouble("spark.task.resource.gpu.amount", -1.0).toFloat

      if (taskGpuAmount == -1.0) {
        // The ETL tasks will not grab a gpu when spark.task.resource.gpu.amount is not set,
        // but with stage-level scheduling, we can make training task grab the gpu.
        return false
      }

      if (taskGpuAmount == executorGpus.toFloat) {
        // spark.executor.resource.gpu.amount = spark.task.resource.gpu.amount
        // results in only 1 task running at a time, which may cause perf issue.
        return true
      }
      // We can enable stage-level scheduling
      false
    } else true // Skip stage-level scheduling for cpu training.
  }

  /**
   * Attempt to modify the task resources so that only one task can be executed
   * on a single executor simultaneously.
   *
   * @param sc  the spark context
   * @param rdd which rdd to be applied with new resource profile
   * @return the original rdd or the changed rdd
   */
  private[spark] def tryStageLevelScheduling(
                                              sc: SparkContext,
                                              xgbExecParams: RuntimeParams,
                                              rdd: RDD[(Booster, Map[String, Array[Float]])]
                                            ): RDD[(Booster, Map[String, Array[Float]])] = {

    val conf = sc.getConf
    if (skipStageLevelScheduling(sc.version, xgbExecParams.runOnGpu, conf)) {
      return rdd
    }

    // Ensure executor_cores is not None
    val executor_cores = conf.getInt("spark.executor.cores", -1)
    if (executor_cores == -1) {
      throw new RuntimeException("Wrong spark.executor.cores")
    }

    // Spark-rapids is a GPU-acceleration project for Spark SQL.
    // When spark-rapids is enabled, we prevent concurrent execution of other ETL tasks
    // that utilize GPUs alongside training tasks in order to avoid GPU out-of-memory errors.
    val spark_plugins = conf.get("spark.plugins", " ")
    val spark_rapids_sql_enabled = conf.get("spark.rapids.sql.enabled", "true")

    // Determine the number of cores required for each task.
    val task_cores = if (spark_plugins.contains("com.nvidia.spark.SQLPlugin") &&
      spark_rapids_sql_enabled.toLowerCase == "true") {
      executor_cores
    } else {
      (executor_cores / 2) + 1
    }

    // Each training task requires cpu cores > total executor cores//2 + 1 to
    // ensure tasks are sent to different executors.
    // Note: We cannot use GPUs to limit concurrent tasks
    // due to https://issues.apache.org/jira/browse/SPARK-45527.
    val task_gpus = 1.0
    val treqs = new TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
    val rp = new ResourceProfileBuilder().require(treqs).build()

    logger.info(s"XGBoost training tasks require the resource(cores=$task_cores, gpu=$task_gpus).")
    rdd.withResources(rp)
  }
}

private[spark] object NewXGBoost extends StageLevelScheduling {
  private val logger = LogFactory.getLog("XGBoostSpark")


  private def trainBooster(rabitEnv: java.util.Map[String, Object],
                           runtimeParams: RuntimeParams,
                           watches: Watches): Booster = {
    val partitionId = TaskContext.getPartitionId()
    val attempt = TaskContext.get().attemptNumber.toString
    rabitEnv.put("DMLC_TASK_ID", partitionId)

    try {
      Communicator.init(rabitEnv)
      val numEarlyStoppingRounds = runtimeParams.earlyStoppingRounds
      val metrics = Array.tabulate(watches.size)(_ =>
        Array.ofDim[Float](runtimeParams.numRounds))

      var params = runtimeParams.toMap

      if (runtimeParams.runOnGpu) {
        val gpuId = if (runtimeParams.isLocal) {
          // For local mode, force gpu id to primary device
          0
        } else {
          getGPUAddrFromResources
        }
        logger.info("Leveraging gpu device " + gpuId + " to train")
        params = params + ("device" -> s"cuda:$gpuId")
      }

      SXGBoost.train(watches.toMap("train"), params, runtimeParams.numRounds,
        watches.toMap, metrics, runtimeParams.obj, runtimeParams.eval,
        earlyStoppingRound = numEarlyStoppingRounds)
    } catch {
      case xgbException: XGBoostError =>
        logger.error(s"XGBooster worker $partitionId has failed $attempt " +
          s"times due to ", xgbException)
        throw xgbException
    } finally {
      Communicator.shutdown()
    }
  }

  /**
   * Train a XGBoost booster with parameters on the dataset
   */
  def train(sc: SparkContext, input: RDD[Watches], params: Map[String, Any]):
  (Booster, Map[String, Array[Float]]) = {
    logger.info(s"Running XGBoost ${spark.VERSION}")

    val paramsFactory = new ParamsFactory(params, sc)
    val runtimeParams = paramsFactory.runtimeParams

    try {
      withTracker(
        runtimeParams.numWorkers,
        runtimeParams.trackerConf
      ) { tracker =>
        val rabitEnv = tracker.getWorkerArgs()

        val boostersAndMetrics = input.barrier().mapPartitions { iter =>
          require(iter.hasNext, "Couldn't get DMatrix")
          val watches = iter.next()

          val metrics = Array.tabulate(watches.size)(_ =>
            Array.ofDim[Float](runtimeParams.numRounds))
          try {
            val booster = trainBooster(rabitEnv, runtimeParams, watches)
            if (TaskContext.getPartitionId() == 0) {
              Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
            } else {
              Iterator.empty
            }
          } finally {
            if (watches != null) {
              watches.delete()
            }
          }
        }

        val rdd = tryStageLevelScheduling(sc, runtimeParams, boostersAndMetrics)
        // The repartition step is to make training stage as ShuffleMapStage, so that when one
        // of the training task fails the training stage can retry. ResultStage won't retry when
        // it fails.
        val (booster, metrics) = rdd.repartition(1).collect()(0)
        (booster, metrics)
      }
    } catch {
      case t: Throwable =>
        // if the job was aborted due to an exception
        logger.error("XGBoost job was aborted due to ", t)
        throw t
    }
  }


  // Executes the provided code block inside a tracker and then stops the tracker
  private def withTracker[T](nWorkers: Int, conf: TrackerConf)(block: ITracker => T): T = {
    val tracker = new RabitTracker(nWorkers, conf.hostIp, conf.port, conf.timeout)
    require(tracker.start(), "FAULT: Failed to start tracker")
    try {
      block(tracker)
    } finally {
      tracker.stop()
    }
  }

}
