% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.Booster.R
\name{print.xgb.Booster}
\alias{print.xgb.Booster}
\title{Print xgb.Booster}
\usage{
\method{print}{xgb.Booster}(x, verbose = FALSE, ...)
}
\arguments{
\item{x}{An \code{xgb.Booster} object.}

\item{verbose}{Whether to print detailed data (e.g., attribute values).}

\item{...}{Not currently used.}
}
\description{
Print information about \code{xgb.Booster}.
}
\examples{
data(agaricus.train, package = "xgboost")
train <- agaricus.train

bst <- xgboost(
  data = train$data,
  label = train$label,
  max_depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 2,
  objective = "binary:logistic"
)

attr(bst, "myattr") <- "memo"

print(bst)
print(bst, verbose = TRUE)

}
