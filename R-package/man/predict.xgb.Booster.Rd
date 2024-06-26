% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.Booster.R
\name{predict.xgb.Booster}
\alias{predict.xgb.Booster}
\alias{predict.xgb.Booster.handle}
\title{Predict method for XGBoost model}
\usage{
\method{predict}{xgb.Booster}(
  object,
  newdata,
  missing = NA,
  outputmargin = FALSE,
  ntreelimit = NULL,
  predleaf = FALSE,
  predcontrib = FALSE,
  approxcontrib = FALSE,
  predinteraction = FALSE,
  reshape = FALSE,
  training = FALSE,
  iterationrange = NULL,
  strict_shape = FALSE,
  ...
)

\method{predict}{xgb.Booster.handle}(object, ...)
}
\arguments{
\item{object}{Object of class \code{xgb.Booster} or \code{xgb.Booster.handle}.}

\item{newdata}{Takes \code{matrix}, \code{dgCMatrix}, \code{dgRMatrix}, \code{dsparseVector},
local data file, or \code{xgb.DMatrix}.
For single-row predictions on sparse data, it is recommended to use the CSR format.
If passing a sparse vector, it will take it as a row vector.}

\item{missing}{Only used when input is a dense matrix. Pick a float value that represents
missing values in data (e.g., 0 or some other extreme value).}

\item{outputmargin}{Whether the prediction should be returned in the form of original untransformed
sum of predictions from boosting iterations' results. E.g., setting \code{outputmargin=TRUE} for
logistic regression would return log-odds instead of probabilities.}

\item{ntreelimit}{Deprecated, use \code{iterationrange} instead.}

\item{predleaf}{Whether to predict pre-tree leaf indices.}

\item{predcontrib}{Whether to return feature contributions to individual predictions (see Details).}

\item{approxcontrib}{Whether to use a fast approximation for feature contributions (see Details).}

\item{predinteraction}{Whether to return contributions of feature interactions to individual predictions (see Details).}

\item{reshape}{Whether to reshape the vector of predictions to matrix form when there are several
prediction outputs per case. No effect if \code{predleaf}, \code{predcontrib},
or \code{predinteraction} is \code{TRUE}.}

\item{training}{Whether the predictions are used for training. For dart booster,
training predicting will perform dropout.}

\item{iterationrange}{Specifies which trees are used in prediction. For
example, take a random forest with 100 rounds.
With \code{iterationrange=c(1, 21)}, only the trees built during \verb{[1, 21)} (half open set)
rounds are used in this prediction. The index is 1-based just like an R vector. When set
to \code{c(1, 1)}, XGBoost will use all trees.}

\item{strict_shape}{Default is \code{FALSE}. When set to \code{TRUE}, the output
type and shape of predictions are invariant to the model type.}

\item{...}{Not used.}
}
\value{
The return type depends on \code{strict_shape}. If \code{FALSE} (default):
\itemize{
\item For regression or binary classification: A vector of length \code{nrows(newdata)}.
\item For multiclass classification: A vector of length \code{num_class * nrows(newdata)} or
a \verb{(nrows(newdata), num_class)} matrix, depending on the \code{reshape} value.
\item When \code{predleaf = TRUE}: A matrix with one column per tree.
\item When \code{predcontrib = TRUE}: When not multiclass, a matrix with
\code{ num_features + 1} columns. The last "+ 1" column corresponds to the baseline value.
In the multiclass case, a list of \code{num_class} such matrices.
The contribution values are on the scale of untransformed margin
(e.g., for binary classification, the values are log-odds deviations from the baseline).
\item When \code{predinteraction = TRUE}: When not multiclass, the output is a 3d array of
dimension \code{c(nrow, num_features + 1, num_features + 1)}. The off-diagonal (in the last two dimensions)
elements represent different feature interaction contributions. The array is symmetric WRT the last
two dimensions. The "+ 1" columns corresponds to the baselines. Summing this array along the last dimension should
produce practically the same result as \code{predcontrib = TRUE}.
In the multiclass case, a list of \code{num_class} such arrays.
}

When \code{strict_shape = TRUE}, the output is always an array:
\itemize{
\item For normal predictions, the output has dimension \verb{(num_class, nrow(newdata))}.
\item For \code{predcontrib = TRUE}, the dimension is \verb{(ncol(newdata) + 1, num_class, nrow(newdata))}.
\item For \code{predinteraction = TRUE}, the dimension is \verb{(ncol(newdata) + 1, ncol(newdata) + 1, num_class, nrow(newdata))}.
\item For \code{predleaf = TRUE}, the dimension is \verb{(n_trees_in_forest, num_class, n_iterations, nrow(newdata))}.
}
}
\description{
Predicted values based on either xgboost model or model handle object.
}
\details{
Note that \code{iterationrange} would currently do nothing for predictions from "gblinear",
since "gblinear" doesn't keep its boosting history.

One possible practical applications of the \code{predleaf} option is to use the model
as a generator of new features which capture non-linearity and interactions,
e.g., as implemented in \code{\link[=xgb.create.features]{xgb.create.features()}}.

Setting \code{predcontrib = TRUE} allows to calculate contributions of each feature to
individual predictions. For "gblinear" booster, feature contributions are simply linear terms
(feature_beta * feature_value). For "gbtree" booster, feature contributions are SHAP
values (Lundberg 2017) that sum to the difference between the expected output
of the model and the current prediction (where the hessian weights are used to compute the expectations).
Setting \code{approxcontrib = TRUE} approximates these values following the idea explained
in \url{http://blog.datadive.net/interpreting-random-forests/}.

With \code{predinteraction = TRUE}, SHAP values of contributions of interaction of each pair of features
are computed. Note that this operation might be rather expensive in terms of compute and memory.
Since it quadratically depends on the number of features, it is recommended to perform selection
of the most important features first. See below about the format of the returned results.

The \code{predict()} method uses as many threads as defined in \code{xgb.Booster} object (all by default).
If you want to change their number, assign a new number to \code{nthread} using \code{\link[=xgb.parameters<-]{xgb.parameters<-()}}.
Note that converting a matrix to \code{\link[=xgb.DMatrix]{xgb.DMatrix()}} uses multiple threads too.
}
\examples{
## binary classification:

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")

## Keep the number of threads to 2 for examples
nthread <- 2
data.table::setDTthreads(nthread)

train <- agaricus.train
test <- agaricus.test

bst <- xgb.train(
  data = xgb.DMatrix(train$data, label = train$label),
  max_depth = 2,
  eta = 0.5,
  nthread = nthread,
  nrounds = 5,
  objective = "binary:logistic"
)

# use all trees by default
pred <- predict(bst, test$data)
# use only the 1st tree
pred1 <- predict(bst, test$data, iterationrange = c(1, 2))

# Predicting tree leafs:
# the result is an nsamples X ntrees matrix
pred_leaf <- predict(bst, test$data, predleaf = TRUE)
str(pred_leaf)

# Predicting feature contributions to predictions:
# the result is an nsamples X (nfeatures + 1) matrix
pred_contr <- predict(bst, test$data, predcontrib = TRUE)
str(pred_contr)
# verify that contributions' sums are equal to log-odds of predictions (up to float precision):
summary(rowSums(pred_contr) - qlogis(pred))
# for the 1st record, let's inspect its features that had non-zero contribution to prediction:
contr1 <- pred_contr[1,]
contr1 <- contr1[-length(contr1)]    # drop BIAS
contr1 <- contr1[contr1 != 0]        # drop non-contributing features
contr1 <- contr1[order(abs(contr1))] # order by contribution magnitude
old_mar <- par("mar")
par(mar = old_mar + c(0,7,0,0))
barplot(contr1, horiz = TRUE, las = 2, xlab = "contribution to prediction in log-odds")
par(mar = old_mar)


## multiclass classification in iris dataset:

lb <- as.numeric(iris$Species) - 1
num_class <- 3

set.seed(11)

bst <- xgb.train(
  data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
  max_depth = 4,
  eta = 0.5,
  nthread = 2,
  nrounds = 10,
  subsample = 0.5,
  objective = "multi:softprob",
  num_class = num_class
)

# predict for softmax returns num_class probability numbers per case:
pred <- predict(bst, as.matrix(iris[, -5]))
str(pred)
# reshape it to a num_class-columns matrix
pred <- matrix(pred, ncol = num_class, byrow = TRUE)
# convert the probabilities to softmax labels
pred_labels <- max.col(pred) - 1
# the following should result in the same error as seen in the last iteration
sum(pred_labels != lb) / length(lb)

# compare with predictions from softmax:
set.seed(11)

bst <- xgb.train(
  data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
  max_depth = 4,
  eta = 0.5,
  nthread = 2,
  nrounds = 10,
  subsample = 0.5,
  objective = "multi:softmax",
  num_class = num_class
)

pred <- predict(bst, as.matrix(iris[, -5]))
str(pred)
all.equal(pred, pred_labels)
# prediction from using only 5 iterations should result
# in the same error as seen in iteration 5:
pred5 <- predict(bst, as.matrix(iris[, -5]), iterationrange = c(1, 6))
sum(pred5 != lb) / length(lb)

}
\references{
\enumerate{
\item Scott M. Lundberg, Su-In Lee, "A Unified Approach to Interpreting Model Predictions",
NIPS Proceedings 2017, \url{https://arxiv.org/abs/1705.07874}
\item Scott M. Lundberg, Su-In Lee, "Consistent feature attribution for tree ensembles",
\url{https://arxiv.org/abs/1706.06060}
}
}
\seealso{
\code{\link[=xgb.train]{xgb.train()}}
}
