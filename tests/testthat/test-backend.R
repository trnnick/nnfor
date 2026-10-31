# Contract every training backend must satisfy (see R/backend.R).
test_that("neuralnet backend satisfies the backend contract", {
  set.seed(1)
  be <- nnfor:::nn_backend("neuralnet")
  X <- matrix(runif(60), 30, dimnames = list(NULL, c("X1", "X2")))
  Y <- matrix(X %*% c(0.3, -0.2), dimnames = list(NULL, "Y"))
  net <- be$train(Y, X, hd = c(3, 2), reps = 2)
  expect_equal(be$name, "neuralnet")
  expect_equal(be$n_members(net), 2)
  expect_length(be$predict(net, X, 2), 30)
  expect_length(be$predict(net, X[1, , drop = FALSE], 1), 1)
  expect_equal(be$layers(net, 1), c(2, 3, 2, 1))
  expect_equal(be$input_names(net), c("X1", "X2"))
  expect_error(nnfor:::nn_backend("nope"), "Unknown network backend")
})

test_that("mlp records its backend", {
  set.seed(1)
  expect_equal(quiet(mlp(AirPassengers, reps = 1))$backend, "neuralnet")
})
