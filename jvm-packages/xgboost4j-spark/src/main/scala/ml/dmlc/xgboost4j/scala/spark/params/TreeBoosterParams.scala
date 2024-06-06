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

import org.apache.spark.ml.param.{BooleanParam, DoubleParam, FloatParam, IntParam, Param, ParamValidators, Params}

/**
 * TreeBoosterParams defines the XGBoost TreeBooster parameters for Spark
 *
 * The details can be found at
 * https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
 */
private[spark] trait TreeBoosterParams extends Params {

  final val eta = new DoubleParam(this, "eta", "Step size shrinkage used in update to prevents " +
    "overfitting. After each boosting step, we can directly get the weights of new features, " +
    "and eta shrinks the feature weights to make the boosting process more conservative.",
    ParamValidators.inRange(0, 1, lowerInclusive = true, upperInclusive = true))

  final def getEta: Double = $(eta)

  def setEta(value: Double): this.type = set(eta, value)

  final val gamma = new DoubleParam(this, "gamma", "Minimum loss reduction required to make a " +
    "further partition on a leaf node of the tree. The larger gamma is, the more conservative " +
    "the algorithm will be.",
    ParamValidators.gtEq(0))

  final def getGamma: Double = $(gamma)

  def setGamma(value: Double): this.type = set(gamma, value)

  final val maxDepth = new IntParam(this, "maxDepth", "Maximum depth of a tree. Increasing this " +
    "value will make the model more complex and more likely to overfit. 0 indicates no limit " +
    "on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. " +
    "exact tree method requires non-zero value.",
    ParamValidators.gtEq(0))

  final def getMaxDepth: Int = $(maxDepth)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  final val minChildWeight = new DoubleParam(this, "minChildWeight", "Minimum sum of instance " +
    "weight (hessian) needed in a child. If the tree partition step results in a leaf node " +
    "with the sum of instance weight less than min_child_weight, then the building process " +
    "will give up further partitioning. In linear regression task, this simply corresponds " +
    "to minimum number of instances needed to be in each node. The larger min_child_weight " +
    "is, the more conservative the algorithm will be.",
    ParamValidators.gtEq(0))

  final def getMinChildWeight: Double = $(minChildWeight)

  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  final val maxDeltaStep = new DoubleParam(this, "maxDeltaStep", "Maximum delta step we allow " +
    "each leaf output to be. If the value is set to 0, it means there is no constraint. If it " +
    "is set to a positive value, it can help making the update step more conservative. Usually " +
    "this parameter is not needed, but it might help in logistic regression when class is " +
    "extremely imbalanced. Set it to value of 1-10 might help control the update.",
    ParamValidators.gtEq(0))

  final def getMaxDeltaStep: Double = $(maxDeltaStep)

  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  final val subsample = new DoubleParam(this, "subsample", "Subsample ratio of the training " +
    "instances. Setting it to 0.5 means that XGBoost would randomly sample half of the " +
    "training data prior to growing trees. and this will prevent overfitting. Subsampling " +
    "will occur once in every boosting iteration.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getSubsample: Double = $(subsample)

  def setSubsample(value: Double): this.type = set(subsample, value)

  final val samplingMethod = new Param[String](this, "samplingMethod", "The method to use to " +
    "sample the training instances. The supported sampling methods" +
    "uniform: each training instance has an equal probability of being selected. Typically set " +
    "subsample >= 0.5 for good results.\n" +
    "gradient_based: the selection probability for each training instance is proportional to " +
    "the regularized absolute value of gradients. subsample may be set to as low as 0.1 " +
    "without loss of model accuracy. Note that this sampling method is only supported when " +
    "tree_method is set to hist and the device is cuda; other tree methods only support " +
    "uniform sampling.",
    ParamValidators.inArray(Array("uniform", "gradient_based")))

  final def getSamplingMethod: String = $(samplingMethod)

  def setSamplingMethod(value: String): this.type = set(samplingMethod, value)

  final val colsampleBytree = new DoubleParam(this, "colsampleBytree", "Subsample ratio of " +
    "columns when constructing each tree. Subsampling occurs once for every tree constructed.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getColsampleBytree: Double = $(colsampleBytree)

  def setColsampleBytree(value: Double): this.type = set(colsampleBytree, value)


  final val colsampleBylevel = new DoubleParam(this, "colsampleBylevel", "Subsample ratio of " +
    "columns for each level. Subsampling occurs once for every new depth level reached in a " +
    "tree. Columns are subsampled from the set of columns chosen for the current tree.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getColsampleBylevel: Double = $(colsampleBylevel)

  def setColsampleBylevel(value: Double): this.type = set(colsampleBylevel, value)

    final val colsampleBynode = new DoubleParam(this, "colsampleBynode", "Subsample ratio of " +
      "columns for each node (split). Subsampling occurs once every time a new split is " +
      "evaluated. Columns are subsampled from the set of columns chosen for the current level.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  final def getColsampleBynode: Double = $(colsampleBynode)

  def setColsampleBynode(value: Double): this.type = set(colsampleBynode, value)

  ////////////////////////////////////////////////////////////////////////////////

  setDefault(eta -> 0.3, gamma -> 0, maxDepth -> 6, minChildWeight -> 1, maxDeltaStep -> 0,
    subsample -> 1, samplingMethod -> "uniform", colsampleBytree -> 1, colsampleBylevel -> 1,
    colsampleBynode->1)

  final val maxLeaves = new IntParam(this, "maxLeaves",
    "Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.",
    (value: Int) => value >= 0)

  final def getMaxLeaves: Int = $(maxLeaves)


  /**
   * L2 regularization term on weights, increase this value will make model more conservative.
   * [default=1]
   */
  final val lambda = new DoubleParam(this, "lambda", "L2 regularization term on weights, " +
    "increase this value will make model more conservative.", (value: Double) => value >= 0)

  final def getLambda: Double = $(lambda)

  /**
   * L1 regularization term on weights, increase this value will make model more conservative.
   * [default=0]
   */
  final val alpha = new DoubleParam(this, "alpha", "L1 regularization term on weights, increase " +
    "this value will make model more conservative.", (value: Double) => value >= 0)

  final def getAlpha: Double = $(alpha)

  /**
   * The tree construction algorithm used in XGBoost. options:
   * {'auto', 'exact', 'approx','gpu_hist'} [default='auto']
   */
  final val treeMethod = new Param[String](this, "treeMethod",
    "The tree construction algorithm used in XGBoost, options: " +
      "{'auto', 'exact', 'approx', 'hist', 'gpu_hist'}",
    (value: String) => BoosterParams.supportedTreeMethods.contains(value))

  final def getTreeMethod: String = $(treeMethod)

  /**
   * The device for running XGBoost algorithms, options: cpu, cuda
   */
  final val device = new Param[String](
    this, "device", "The device for running XGBoost algorithms, options: cpu, cuda",
    (value: String) => BoosterParams.supportedDevices.contains(value)
  )

  final def getDevice: String = $(device)

  /**
   * growth policy for fast histogram algorithm
   */
  final val growPolicy = new Param[String](this, "growPolicy",
    "Controls a way new nodes are added to the tree. Currently supported only if" +
      " tree_method is set to hist. Choices: depthwise, lossguide. depthwise: split at nodes" +
      " closest to the root. lossguide: split at nodes with highest loss change.",
    (value: String) => BoosterParams.supportedGrowthPolicies.contains(value))

  final def getGrowPolicy: String = $(growPolicy)

  /**
   * maximum number of bins in histogram
   */
  final val maxBins = new IntParam(this, "maxBin", "maximum number of bins in histogram",
    (value: Int) => value > 0)

  final def getMaxBins: Int = $(maxBins)

  /**
   * whether to build histograms using single precision floating point values
   */
  final val singlePrecisionHistogram = new BooleanParam(this, "singlePrecisionHistogram",
    "whether to use single precision to build histograms")

  final def getSinglePrecisionHistogram: Boolean = $(singlePrecisionHistogram)

  /**
   * Control the balance of positive and negative weights, useful for unbalanced classes. A typical
   * value to consider: sum(negative cases) / sum(positive cases).   [default=1]
   */
  final val scalePosWeight = new DoubleParam(this, "scalePosWeight", "Control the balance of " +
    "positive and negative weights, useful for unbalanced classes. A typical value to consider:" +
    " sum(negative cases) / sum(positive cases)")

  final def getScalePosWeight: Double = $(scalePosWeight)

  // Dart boosters

  /**
   * Parameter for Dart booster.
   * Type of sampling algorithm. "uniform": dropped trees are selected uniformly.
   * "weighted": dropped trees are selected in proportion to weight. [default="uniform"]
   */
  final val sampleType = new Param[String](this, "sampleType", "type of sampling algorithm, " +
    "options: {'uniform', 'weighted'}",
    (value: String) => BoosterParams.supportedSampleType.contains(value))

  final def getSampleType: String = $(sampleType)

  /**
   * Parameter of Dart booster.
   * type of normalization algorithm, options: {'tree', 'forest'}. [default="tree"]
   */
  final val normalizeType = new Param[String](this, "normalizeType", "type of normalization" +
    " algorithm, options: {'tree', 'forest'}",
    (value: String) => BoosterParams.supportedNormalizeType.contains(value))

  final def getNormalizeType: String = $(normalizeType)

  /**
   * Parameter of Dart booster.
   * dropout rate. [default=0.0] range: [0.0, 1.0]
   */
  final val rateDrop = new DoubleParam(this, "rateDrop", "dropout rate", (value: Double) =>
    value >= 0 && value <= 1)

  final def getRateDrop: Double = $(rateDrop)

  /**
   * Parameter of Dart booster.
   * probability of skip dropout. If a dropout is skipped, new trees are added in the same manner
   * as gbtree. [default=0.0] range: [0.0, 1.0]
   */
  final val skipDrop = new DoubleParam(this, "skipDrop", "probability of skip dropout. If" +
    " a dropout is skipped, new trees are added in the same manner as gbtree.",
    (value: Double) => value >= 0 && value <= 1)

  final def getSkipDrop: Double = $(skipDrop)

  // linear booster
  /**
   * Parameter of linear booster
   * L2 regularization term on bias, default 0(no L1 reg on bias because it is not important)
   */
  final val lambdaBias = new DoubleParam(this, "lambdaBias", "L2 regularization term on bias, " +
    "default 0 (no L1 reg on bias because it is not important)", (value: Double) => value >= 0)

  final def getLambdaBias: Double = $(lambdaBias)

  final val treeLimit = new IntParam(this, name = "treeLimit",
    doc = "number of trees used in the prediction; defaults to 0 (use all trees).")
  setDefault(treeLimit, 0)

  final def getTreeLimit: Int = $(treeLimit)

  final val monotoneConstraints = new Param[String](this, name = "monotoneConstraints",
    doc = "a list in length of number of features, 1 indicate monotonic increasing, - 1 means " +
      "decreasing, 0 means no constraint. If it is shorter than number of features, 0 will be " +
      "padded ")

  final def getMonotoneConstraints: String = $(monotoneConstraints)

  final val interactionConstraints = new Param[String](this,
    name = "interactionConstraints",
    doc = "Constraints for interaction representing permitted interactions. The constraints" +
      " must be specified in the form of a nest list, e.g. [[0, 1], [2, 3, 4]]," +
      " where each inner list is a group of indices of features that are allowed to interact" +
      " with each other. See tutorial for more information")

  final def getInteractionConstraints: String = $(interactionConstraints)
}
