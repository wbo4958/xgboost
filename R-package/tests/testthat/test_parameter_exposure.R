context('Test model params and call are exposed to R')

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')

dtrain <- xgb.DMatrix(
  agaricus.train$data, label = agaricus.train$label, nthread = 2
)
dtest <- xgb.DMatrix(
  agaricus.test$data, label = agaricus.test$label, nthread = 2
)

bst <- xgb.train(data = dtrain,
                 max_depth = 2,
                 eta = 1,
                 nrounds = 10,
                 nthread = 1,
                 verbose = 0,
                 objective = "binary:logistic")

test_that("call is exposed to R", {
  expect_false(is.null(bst$call))
  expect_is(bst$call, "call")
})

test_that("params is exposed to R", {
  model_params <- bst$params
  expect_is(model_params, "list")
  expect_equal(model_params$eta, 1)
  expect_equal(model_params$max_depth, 2)
  expect_equal(model_params$objective, "binary:logistic")
})
