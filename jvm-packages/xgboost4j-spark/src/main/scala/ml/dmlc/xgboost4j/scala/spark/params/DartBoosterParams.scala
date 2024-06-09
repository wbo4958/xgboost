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

package ml.dmlc.xgboost4j.scala.spark.params

import org.apache.spark.ml.param._

private[spark] trait DartBoosterParams extends Params {

  final val sampleType = new Param[String](this, "sampleType", "Type of sampling algorithm, " +
    "options: {'uniform', 'weighted'}", ParamValidators.inArray(Array("uniform", "weighted")))

  final def getSampleType: String = $(sampleType)

  final val normalizeType = new Param[String](this, "normalizeType", "type of normalization" +
    " algorithm, options: {'tree', 'forest'}",
    (value: String) => BoosterParams.supportedNormalizeType.contains(value))

  final def getNormalizeType: String = $(normalizeType)

  final val rateDrop = new DoubleParam(this, "rateDrop", "Dropout rate (a fraction of previous " +
    "trees to drop during the dropout)",
    ParamValidators.inRange(0, 1, true, true))

  final def getRateDrop: Double = $(rateDrop)

  final val oneDrop = new Param[Boolean](this, "oneDrop", "When this flag is enabled, at least " +
    "one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout " +
    "from the original DART paper)")

  final def getOneDrop: Boolean = $(oneDrop)

  final val skipDrop = new DoubleParam(this, "skipDrop", "Probability of skipping the dropout " +
    "procedure during a boosting iteration.\nIf a dropout is skipped, new trees are added " +
    "in the same manner as gbtree.\nNote that non-zero skip_drop has higher priority than " +
    "rate_drop or one_drop.",
    ParamValidators.inRange(0, 0, true, true))

  final def getSkipDrop: Double = $(skipDrop)

  setDefault(sampleType -> "uniform", normalizeType -> "tree", rateDrop -> 0, skipDrop -> 0)

}
