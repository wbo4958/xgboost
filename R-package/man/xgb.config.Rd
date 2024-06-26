% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.Booster.R
\name{xgb.config}
\alias{xgb.config}
\alias{xgb.config<-}
\title{Accessors for model parameters as JSON string}
\usage{
xgb.config(object)

xgb.config(object) <- value
}
\arguments{
\item{object}{Object of class \code{xgb.Booster}.}

\item{value}{A JSON string.}
}
\description{
Accessors for model parameters as JSON string
}
\examples{
data(agaricus.train, package = "xgboost")

## Keep the number of threads to 1 for examples
nthread <- 1
data.table::setDTthreads(nthread)
train <- agaricus.train

bst <- xgboost(
  data = train$data,
  label = train$label,
  max_depth = 2,
  eta = 1,
  nthread = nthread,
  nrounds = 2,
  objective = "binary:logistic"
)

config <- xgb.config(bst)

}
