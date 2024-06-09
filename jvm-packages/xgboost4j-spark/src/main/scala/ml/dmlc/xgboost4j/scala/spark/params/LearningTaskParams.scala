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

import org.apache.spark.ml.param._

import scala.collection.immutable.HashSet

/**
 * Specify the learning task and the corresponding learning objective.
 * More details can be found at https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
 */
private[spark] trait LearningTaskParams extends Params {


  final val objective = new Param[String](this, "objective",
    "Objective function used for training",
    ParamValidators.inArray(LearningTaskParams.supportedObjectives.toArray))

  final def getObjective: String = $(objective)

  final val baseScore = new DoubleParam(this, "baseScore", "The initial prediction score of all " +
    "instances, global bias. The parameter is automatically estimated for selected objectives " +
    "before training. To disable the estimation, specify a real number argument. For sufficient " +
    "number of iterations, changing this value will not have too much effect.")

  final def getBaseScore: Double = $(baseScore)

  final val evalMetric = new Param[String](this, "evalMetric", "Evaluation metrics for " +
    "validation data, a default metric will be assigned according to objective (rmse for " +
    "regression, and logloss for classification, mean average precision for rank:map, etc.)" +
    "User can add multiple evaluation metrics. Python users: remember to pass the metrics in " +
    "as list of parameters pairs instead of map, so that latter eval_metric won't override " +
    "previous ones", ParamValidators.inArray(LearningTaskParams.supportedEvalMetrics.toArray))

  final def getEvalMetric: String = $(evalMetric)

  final val seed = new LongParam(this, "seed", "Random number seed.")

  final def getSeed: Long = $(seed)

  final val seedPerIteration = new Param[Boolean](this, "seed_per_iteration", "Seed PRNG " +
    "determnisticly via iterator number..")

  final def getSeedPerIteration: Boolean = $(seedPerIteration)

  // Parameters for Tweedie Regression (objective=reg:tweedie)
  final val tweedieVariancePower = new DoubleParam(this, "tweedie_variance_power", "Parameter " +
    "that controls the variance of the Tweedie distribution var(y) ~ E(y)^tweedie_variance_power.",
    ParamValidators.inRange(1, 2, false, false))

  final def getTweedieVariancePower: Double = $(tweedieVariancePower)

  // Parameter for using Pseudo-Huber (reg:pseudohubererror)
  final val huberSlope = new DoubleParam(this, "huber_slope", "A parameter used for Pseudo-Huber " +
    "loss to define the (delta) term.")

  final def getHuberSlope: Double = $(huberSlope)

  // Parameter for using Quantile Loss (reg:quantileerror) TODO

  // Parameter for using AFT Survival Loss (survival:aft) and Negative
  // Log Likelihood of AFT metric (aft-nloglik)
  final val aftLossDistribution = new Param[String](this, "aftLossDistribution", "Probability " +
    "Density Function",
    ParamValidators.inArray(Array("normal", "logistic", "extreme")))

  final def getAftLossDistribution: String = $(aftLossDistribution)

  // Parameters for learning to rank (rank:ndcg, rank:map, rank:pairwise)
  final val lambdarankPairMethod = new Param[String](this, "lambdarankPairMethod", "pairs for " +
    "pair-wise learning",
    ParamValidators.inArray(Array("mean", "topk")))

  final def getLambdarankPairMethod: String = $(aftLossDistribution)

  final val lambdarankNumPairPerSample = new IntParam(this, "lambdarank_num_pair_per_sample",
    "It specifies the number of pairs sampled for each document when pair method is mean, or" +
      " the truncation level for queries when the pair method is topk. For example, to train " +
      "with ndcg@6, set lambdarank_num_pair_per_sample to 6 and lambdarank_pair_method to topk",
    ParamValidators.gtEq(1))

  final def getLambdarankNumPairPerSample: Double = $(lambdarankNumPairPerSample)

  setDefault(objective -> "reg:squarederror", seed -> 0, seedPerIteration -> false,
    tweedieVariancePower -> 1.5, huberSlope -> 1, lambdarankPairMethod -> "mean")

}

private[spark] object LearningTaskParams {

  val supportedObjectives = HashSet("reg:squarederror", "reg:squaredlogerror", "reg:logistic",
    "reg:pseudohubererror", "reg:absoluteerror", "reg:quantileerror", "binary:logistic",
    "binary:logitraw", "binary:hinge", "count:poisson", "survival:cox", "survival:aft",
    "multi:softmax", "multi:softprob", "rank:ndcg", "rank:map", "rank:pairwise", "reg:gamma",
    "reg:tweedie")

  val supportedEvalMetrics = HashSet("rmse", "rmsle", "mae", "mape", "mphe", "logloss", "error",
    "error@t", "merror", "mlogloss", "auc", "aucpr", "pre", "ndcg", "map", "ndcg@n", "map@n",
    "pre@n", "ndcg-", "map-", "ndcg@n-", "map@n-", "poisson-nloglik", "gamma-nloglik",
    "cox-nloglik", "gamma-deviance", "tweedie-nloglik", "aft-nloglik",
    "interval-regression-accuracy")

}
