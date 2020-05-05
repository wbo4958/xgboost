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

  def transform(clsifier: XGBoostClassifier, schema: StructType): StructType = {
    schema
  }

  /**
   * give plugin more control for the parameters
   *
   * @param params
   * @param hasGroup
   */
  def initialize(
      params: Map[String, Any],
      rawParams: Map[String, Any],
      hasGroup: Boolean = false,
      hasEvals: Boolean,
      sc: SparkContext)

  /**
   * convert dataset to any kind of RDD
   *
   * @param dataset
   * @return
   */
  def convertDatasetToRdd(
      est: Estimator[_],
      params: Map[String, Any],
      dataset: Dataset[_],
      evalSetsMap: Map[String, Dataset[_]] = Map.empty): RDD[_]

  /**
   * This method runs on executor side, since iter is the type erasure, you can cast iter into
   * any iterator type
   *
   * @param iter
   * @return
   */
  def buildWatches(iter: Iterator[_]): Watches

  def onPreRabitInit: Unit = {}

  def onPostRabitInit: Unit = {}

  def onPreXGboostTrain: Unit = {}

  def onPostXGBoostTrain: Unit = {}

  def onExecutorError(err: XGBoostError): Unit = {}

  def onExecutorCleanUp: Unit = {}

  def onDriverCleanUp: Unit = {}

  /**
   * give the implementation more control on xgboost params
   *
   * @param params
   * @return
   */
  def getFinalXGBoostParam(params: Map[String, Any]): Map[String, Any]

}
