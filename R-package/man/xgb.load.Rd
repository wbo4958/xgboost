% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.load.R
\name{xgb.load}
\alias{xgb.load}
\title{Load xgboost model from binary file}
\usage{
xgb.load(modelfile)
}
\arguments{
\item{modelfile}{the name of the binary input file.}
}
\value{
An object of \code{xgb.Booster} class.
}
\description{
Load xgboost model from the binary model file.
}
\details{
The input file is expected to contain a model saved in an xgboost model format
using either \code{\link{xgb.save}} or \code{\link{cb.save.model}} in R, or using some
appropriate methods from other xgboost interfaces. E.g., a model trained in Python and
saved from there in xgboost format, could be loaded from R.

Note: a model saved as an R-object, has to be loaded using corresponding R-methods,
not \code{xgb.load}.
}
\examples{
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')

## Keep the number of threads to 1 for examples
nthread <- 1
data.table::setDTthreads(nthread)

train <- agaricus.train
test <- agaricus.test
bst <- xgb.train(
  data = xgb.DMatrix(train$data, label = train$label),
  max_depth = 2,
  eta = 1,
  nthread = nthread,
  nrounds = 2,
  objective = "binary:logistic"
)

xgb.save(bst, 'xgb.model')
bst <- xgb.load('xgb.model')
if (file.exists('xgb.model')) file.remove('xgb.model')
}
\seealso{
\code{\link{xgb.save}}, \code{\link{xgb.Booster.complete}}.
}
