/*
 Copyright (c) 2014-2024 by Contributors

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

package ml.dmlc.xgboost4j.scala.spark.params

import com.google.common.base.CaseFormat
import ml.dmlc.xgboost4j.scala.spark.TrackerConf
import org.apache.spark.ml.param._

import scala.collection.mutable

/**
 * General xgboost parameters, more details can be found
 * at https://xgboost.readthedocs.io/en/stable/parameter.html#general-parameters
 */
private[spark] trait GeneralParams extends Params {

  final val booster = new Param[String](this, "booster", "Which booster to use. Can be gbtree, " +
    "gblinear or dart; gbtree and dart use tree based models while gblinear uses linear " +
    "functions.", ParamValidators.inArray(Array("gbtree", "dart")))

  final def getBooster: String = $(booster)

  final val device = new Param[String](this, "device", "Device for XGBoost to run. User can " +
    "set it to one of the following values: {cpu, cuda, gpu}",
    ParamValidators.inArray(Array("cpu", "cuda", "gpu")))

  final def getDevice: String = $(device)

  final val verbosity = new IntParam(this, "verbosity", "Verbosity of printing messages. Valid " +
    "values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change " +
    "configurations based on heuristics, which is displayed as warning message. If there's " +
    "unexpected behaviour, please try to increase value of verbosity.",
    ParamValidators.inRange(0, 3, true, true))

  final def getVerbosity: Int = $(verbosity)

  final val nthread = new IntParam(this, "nthread", "Number of threads used by per worker",
    ParamValidators.gtEq(1))

  final def getNthread: Int = $(nthread)

  setDefault(booster -> "gbtree", device -> "cpu", verbosity -> 1, nthread->1)

  /**
   * The number of rounds for boosting
   */


  /**
   * number of workers used to train xgboost model. default: 1
   */
  final val numWorkers = new IntParam(this, "numWorkers", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))
  setDefault(numWorkers, 1)

  final def getNumWorkers: Int = $(numWorkers)

  /**
   * number of threads used by per worker. default 1
   */


  /**
   * whether to use external memory as cache. default: false
   */
  final val useExternalMemory = new BooleanParam(this, "useExternalMemory",
    "whether to use external memory as cache")
  setDefault(useExternalMemory, false)

  final def getUseExternalMemory: Boolean = $(useExternalMemory)

  /**
   * Deprecated. Please use verbosity instead.
   * 0 means printing running messages, 1 means silent mode. default: 0
   */
  final val silent = new IntParam(this, "silent",
    "Deprecated. Please use verbosity instead. " +
      "0 means printing running messages, 1 means silent mode.",
    (value: Int) => value >= 0 && value <= 1)

  final def getSilent: Int = $(silent)

  /**
   * Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
   * default: 1
   */


  /**
   * customized objective function provided by user. default: null
   */
  final val customObj = new CustomObjParam(this, "customObj", "customized objective function " +
    "provided by user")

  /**
   * customized evaluation function provided by user. default: null
   */
  final val customEval = new CustomEvalParam(this, "customEval",
    "customized evaluation function provided by user")

  /**
   * Rabit tracker configurations. The parameter must be provided as an instance of the
   * TrackerConf class, which has the following definition:
   *
   * case class TrackerConf(timeout: Int, hostIp: String, port: Int)
   *
   * See below for detailed explanations.
   *
   *   - timeout : The maximum wait time for all workers to connect to the tracker. (in seconds)
   *     default: 0 (no timeout)
   *
   * Timeout for constructing the communication group and waiting for the tracker to
   * shutdown when it's instructed to, doesn't apply to communication when tracking
   * is running.
   * The timeout value should take the time of data loading and pre-processing into account,
   * due to potential lazy execution. Alternatively, you may force Spark to
   * perform data transformation before calling XGBoost.train(), so that this timeout truly
   * reflects the connection delay. Set a reasonable timeout value to prevent model
   * training/testing from hanging indefinitely, possible due to network issues.
   * Note that zero timeout value means to wait indefinitely (equivalent to Duration.Inf).
   *
   *   - hostIp : The Rabit Tracker host IP address. This is only needed if the host IP
   *     cannot be automatically guessed.
   *
   *   - port : The port number for the tracker to listen to. Use a system allocated one by
   *     default.
   */
  final val trackerConf = new TrackerConfParam(this, "trackerConf", "Rabit tracker configurations")
  setDefault(trackerConf, TrackerConf())


  /** Feature's name, it will be set to DMatrix and Booster, and in the final native json model.
   * In native code, the parameter name is feature_name.
   * */
  final val featureNames = new StringArrayParam(this, "feature_names",
    "an array of feature names")

  final def getFeatureNames: Array[String] = $(featureNames)

  /** Feature types, q is numeric and c is categorical.
   * In native code, the parameter name is feature_type
   * */
  final val featureTypes = new StringArrayParam(this, "feature_types",
    "an array of feature types")

  final def getFeatureTypes: Array[String] = $(featureTypes)
}


private[spark] trait ParamMapFuncs extends Params {

  def XGBoost2MLlibParams(xgboostParams: Map[String, Any]): Unit = {
    for ((paramName, paramValue) <- xgboostParams) {
      if ((paramName == "booster" && paramValue != "gbtree") ||
        (paramName == "updater" && paramValue != "grow_histmaker,prune" &&
          paramValue != "grow_quantile_histmaker" && paramValue != "grow_gpu_hist")) {
        throw new IllegalArgumentException(s"you specified $paramName as $paramValue," +
          s" XGBoost-Spark only supports gbtree as booster type and grow_histmaker or" +
          s" grow_quantile_histmaker or grow_gpu_hist as the updater type")
      }
      val name = CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, paramName)
      params.find(_.name == name).foreach {
        case _: DoubleParam =>
          set(name, paramValue.toString.toDouble)
        case _: BooleanParam =>
          set(name, paramValue.toString.toBoolean)
        case _: IntParam =>
          set(name, paramValue.toString.toInt)
        case _: FloatParam =>
          set(name, paramValue.toString.toFloat)
        case _: LongParam =>
          set(name, paramValue.toString.toLong)
        case _: Param[_] =>
          set(name, paramValue)
      }
    }
  }

  def MLlib2XGBoostParams: Map[String, Any] = {
    val xgboostParams = new mutable.HashMap[String, Any]()
    for (param <- params) {
      if (isDefined(param)) {
        val name = CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, param.name)
        xgboostParams += name -> $(param)
      }
    }
    xgboostParams.toMap
  }
}
