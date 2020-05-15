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

import ml.dmlc.xgboost4j.scala.spark.train._
import org.apache.spark.ml.Estimator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, Dataset}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.commons.logging.LogFactory
import org.apache.spark.sql.types.StructType
import org.apache.spark.{SparkContext}

private[spark] class LabeledPointTrain extends TrainPlugin {

  private val logger = LogFactory.getLog("LabeledPointTrain")

  private var hasGroup: Boolean = false

  private var xgbExecParams: XGBoostExecutionParams = _

  private var labeledPointImpl: LabeledPointImplIntf = _


  /**
   * plugin implements its own transform on schema
   *
   * @param est
   * @param schema
   * @return
   */
  override def transformSchema(est: Estimator[_], schema: StructType, logging: Boolean):
      StructType = {
    est.transformSchema(schema)
  }

  /**
   * initialize will be called first
   *
   * @param sc SparkContext
   * @param params
   * @param hasGroup
   * @param hasEvalSets
   */
  override def initialize(sc: SparkContext, params: Map[String, Any], hasGroup: Boolean,
      hasEvalSets: Boolean): Unit = {
    this.hasGroup = hasGroup
    xgbExecParams = new XGBoostExecutionParamsFactory(params, sc).buildXGBRuntimeParams
    labeledPointImpl = getLabeledPointImpl(hasGroup, hasEvalSets, xgbExecParams)
  }


  /**
   * Convert dataset to any kind of RDD which will be used to build Watches later
   *
   * @param est
   * @param params
   * @param dataset
   * @param evalSetsMap
   * @return
   */
  override def extractRdd(est: Estimator[_], params: Map[String, Any], dataset: Dataset[_],
      evalSetsMap: Map[String, Dataset[_]]): RDD[_] = {

    val numWorkers = xgbExecParams.numWorkers
    val (labelCol, featuresCol, weight, baseMargin, group, deterministicPartition) =
      est match {
        case clsifier: XGBoostClassifier => {
          clsifier.transformSchema(dataset.schema)
          val weight = if (!clsifier.isDefined(clsifier.weightCol) ||
            clsifier.getWeightCol.isEmpty) {
            lit(1.0)
          } else col(clsifier.getWeightCol)

          val baseMargin = if (!clsifier.isDefined(clsifier.baseMarginCol) ||
            clsifier.getBaseMarginCol.isEmpty) {
            lit(Float.NaN)
          } else col(clsifier.getWeightCol)

          (col(clsifier.getLabelCol), col(clsifier.getFeaturesCol), weight, baseMargin,
            None, clsifier.needDeterministicRepartitioning)
        }
        case regressor: XGBoostRegressor => throw new RuntimeException(
          "regressor not support for now")
      }

    val trainingSet: RDD[XGBLabeledPoint] = DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      labelCol, featuresCol, weight, baseMargin, None, numWorkers, deterministicPartition,
      dataset.asInstanceOf[DataFrame]).head

    val evalRDDMap = evalSetsMap.map {
      case (name, dataFrame) => (name,
        DataUtils.convertDataFrameToXGBLabeledPointRDDs(labelCol, featuresCol, weight, baseMargin,
          None, numWorkers, deterministicPartition, dataFrame.toDF()).head)
    }

    labeledPointImpl.rddTransform(trainingSet, numWorkers, evalRDDMap)
  }

  /**
   * This method runs on executor side, since iter is the type erasure, you can cast iter into
   * any iterator type
   *
   * @param iter
   * @return
   */
  override def buildWatches(iter: Iterator[_]): Watches = {
    labeledPointImpl.buildWatches(iter)
  }

  /**
   * give the implementation more control on xgboost params
   *
   * @param params
   * @return
   */
  override def getFinalXGBoostParam(params: Map[String, Any]): Map[String, Any] = {
    params
  }

  override def onDriverCleanUp: Unit = {
    labeledPointImpl.cleanUp
  }

  private def getLabeledPointImpl(
      hasGroup: Boolean, hasEvals: Boolean,
      xgbParams: XGBoostExecutionParams): LabeledPointImplIntf = {

    (hasGroup, hasEvals) match {
      case (true, true) => TrainForRankWithEval(xgbParams.cacheTrainingSet, xgbParams.missing,
        xgbParams.allowNonZeroForMissing, xgbParams.useExternalMemory)

      case (true, false) => TrainForRankWithoutEval(xgbParams.cacheTrainingSet, xgbParams.missing,
        xgbParams.allowNonZeroForMissing, xgbParams.useExternalMemory,
        xgbParams.xgbInputParams.trainTestRatio, xgbParams.xgbInputParams.seed)

      case (false, true) => TrainForNonRankWithEval(xgbParams.cacheTrainingSet, xgbParams.missing,
        xgbParams.allowNonZeroForMissing, xgbParams.useExternalMemory)

      case _ => TrainForNonRankWithoutEval(xgbParams.cacheTrainingSet, xgbParams.missing,
        xgbParams.allowNonZeroForMissing, xgbParams.useExternalMemory,
        xgbParams.xgbInputParams.trainTestRatio, xgbParams.xgbInputParams.seed)
    }
  }
}
