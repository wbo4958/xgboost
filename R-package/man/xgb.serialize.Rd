% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.serialize.R
\name{xgb.serialize}
\alias{xgb.serialize}
\title{Serialize the booster instance into R's raw vector.  The serialization method differs
from \code{\link{xgb.save.raw}} as the latter one saves only the model but not
parameters.  This serialization format is not stable across different xgboost versions.}
\usage{
xgb.serialize(booster)
}
\arguments{
\item{booster}{the booster instance}
}
\description{
Serialize the booster instance into R's raw vector.  The serialization method differs
from \code{\link{xgb.save.raw}} as the latter one saves only the model but not
parameters.  This serialization format is not stable across different xgboost versions.
}
\examples{
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
bst <- xgb.train(data = xgb.DMatrix(train$data, label = train$label), max_depth = 2,
                 eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
raw <- xgb.serialize(bst)
bst <- xgb.unserialize(raw)

}
