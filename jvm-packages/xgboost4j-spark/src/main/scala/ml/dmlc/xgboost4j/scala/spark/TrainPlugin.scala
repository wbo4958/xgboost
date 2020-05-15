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

import ml.dmlc.xgboost4j.java.XGBoostError
import org.apache.spark.SparkContext
import org.apache.spark.ml.Estimator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

trait TrainPlugin {

  /**
   * plugin implements its own transform on schema
   * @param est
   * @param schema
   * @return
   */
  def transformSchema(est: Estimator[_], schema: StructType, logging: Boolean = false):
      StructType = {
    schema
  }

  /**
   * initialize will be called first
   * @param sc SparkContext
   * @param params
   * @param hasGroup
   * @param hasEvalSets
   */
  def initialize(
      sc: SparkContext,
      params: Map[String, Any],
      hasGroup: Boolean = false,
      hasEvalSets: Boolean = false): Unit = {}

  /**
   * Convert dataset to any kind of RDD which will be used to build Watches later
   * @param est
   * @param params
   * @param dataset
   * @param evalSetsMap
   * @return
   */
  def extractRdd(
      est: Estimator[_],
      params: Map[String, Any],
      dataset: Dataset[_],
      evalSetsMap: Map[String, Dataset[_]] = Map.empty): RDD[_]

  /**
   * This method runs on executor side, since iter is the type erasure, you can cast it into
   * any iterator type
   *
   * @param iter
   * @return
   */
  def buildWatches(iter: Iterator[_]): Watches

  /**
   * allow plugin modify the parameters which will be passed into XGBoost lib
   *
   * @param params
   * @return
   */
  def getFinalXGBoostParam(params: Map[String, Any]): Map[String, Any] = {
    params
  }

  /**
   * called before rabit init in executor side
   */
  def onPreRabitInit: Unit = {}

  /**
   * called after rabit init in executor side
   */
  def onPostRabitInit: Unit = {}

  /**
   * called before xgboost lib train in executor side
   */
  def onPreXGboostTrain: Unit = {}

  /**
   * called after xgboost lib train in executor side
   */
  def onPostXGBoostTrain: Unit = {}

  /**
   * Error occurred in executor side
   * @param err
   */
  def onExecutorError(err: XGBoostError): Unit = {}

  /**
   * clean up in executor side
   */
  def onExecutorCleanUp: Unit = {}

  /**
   * clean up in driver side
   */
  def onDriverCleanUp: Unit = {}

}
