% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.DMatrix.R
\name{xgb.DMatrix}
\alias{xgb.DMatrix}
\title{Construct xgb.DMatrix object}
\usage{
xgb.DMatrix(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  missing = NA,
  silent = FALSE,
  feature_names = colnames(data),
  nthread = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL,
  enable_categorical = FALSE
)
}
\arguments{
\item{data}{a \code{matrix} object (either numeric or integer), a \code{dgCMatrix} object,
a \code{dgRMatrix} object,
a \code{dsparseVector} object (only when making predictions from a fitted model, will be
interpreted as a row vector), or a character string representing a filename.}

\item{label}{Label of the training data.}

\item{weight}{Weight for each instance.

Note that, for ranking task, weights are per-group.  In ranking task, one weight
is assigned to each group (not each data point). This is because we
only care about the relative ordering of data points within each group,
so it doesn't make sense to assign weights to individual data points.}

\item{base_margin}{Base margin used for boosting from existing model.

\if{html}{\out{<div class="sourceCode">}}\preformatted{   In the case of multi-output models, one can also pass multi-dimensional base_margin.
}\if{html}{\out{</div>}}}

\item{missing}{a float value to represents missing values in data (used only when input is a dense matrix).
It is useful when a 0 or some other extreme value represents missing values in data.}

\item{silent}{whether to suppress printing an informational message after loading from a file.}

\item{feature_names}{Set names for features. Overrides column names in data
frame and matrix.}

\item{nthread}{Number of threads used for creating DMatrix.}

\item{group}{Group size for all ranking group.}

\item{qid}{Query ID for data samples, used for ranking.}

\item{label_lower_bound}{Lower bound for survival training.}

\item{label_upper_bound}{Upper bound for survival training.}

\item{feature_weights}{Set feature weights for column sampling.}

\item{enable_categorical}{Experimental support of specializing for categorical features.

\if{html}{\out{<div class="sourceCode">}}\preformatted{                      If passing 'TRUE' and 'data' is a data frame,
                      columns of categorical types will automatically
                      be set to be of categorical type (feature_type='c') in the resulting DMatrix.

                      If passing 'FALSE' and 'data' is a data frame with categorical columns,
                      it will result in an error being thrown.

                      If 'data' is not a data frame, this argument is ignored.

                      JSON/UBJSON serialization format is required for this.
}\if{html}{\out{</div>}}}
}
\description{
Construct xgb.DMatrix object from either a dense matrix, a sparse matrix, or a local file.
Supported input file formats are either a LIBSVM text file or a binary file that was created previously by
\code{\link{xgb.DMatrix.save}}).
}
\details{
Note that DMatrix objects are not serializable through R functions such as \code{saveRDS} or \code{save}.
If a DMatrix gets serialized and then de-serialized (for example, when saving data in an R session or caching
chunks in an Rmd file), the resulting object will not be usable anymore and will need to be reconstructed
from the original source of data.
}
\examples{
data(agaricus.train, package='xgboost')
## Keep the number of threads to 1 for examples
nthread <- 1
data.table::setDTthreads(nthread)
dtrain <- with(
  agaricus.train, xgb.DMatrix(data, label = label, nthread = nthread)
)
xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
dtrain <- xgb.DMatrix('xgb.DMatrix.data')
if (file.exists('xgb.DMatrix.data')) file.remove('xgb.DMatrix.data')
}
