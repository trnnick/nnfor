skip_if_no_torch <- function() {
  skip_if_not(nnfor:::torch_available(), "torch is not installed")
}

short_y <- function() window(AirPassengers, end = c(1956, 12))

test_that("torch backend satisfies the backend contract", {
  skip_if_no_torch()
  set.seed(1)
  be <- nnfor:::nn_backend("torch")
  X <- matrix(runif(60), 30, dimnames = list(NULL, c("X1", "X2")))
  Y <- matrix(X %*% c(0.3, -0.2), dimnames = list(NULL, "Y"))
  net <- be$train(Y, X, hd = c(3, 2), reps = 2, stepmax = 200)
  expect_equal(be$name, "torch")
  expect_equal(be$n_members(net), 2)
  expect_length(be$predict(net, X, 2), 30)
  expect_length(be$predict(net, X[1, , drop = FALSE], 1), 1)
  expect_equal(be$layers(net, 1), c(2, 3, 2, 1))
  expect_equal(be$input_names(net), c("X1", "X2"))
})

test_that("members trained together equal members trained alone", {
  skip_if_no_torch()
  X <- matrix(runif(80), 40, dimnames = list(NULL, c("X1", "X2")))
  Y <- matrix(sin(3 * X[, 1]) - X[, 2], dimnames = list(NULL, "Y"))
  x <- torch::torch_tensor(X); y <- torch::torch_tensor(Y)
  act <- nnfor:::torch_activation("logistic")
  set.seed(5)
  sw <- matrix(rnorm(nnfor:::neuralnet_n_weights(2, 4) * 3), nrow = 3, byrow = TRUE)
  for (alg in c("rprop", "adam")) {
    settings <- list(algorithm = alg, threshold = 0.01, stepmax = 300, learningrate = NULL)
    rp <- nnfor:::rprop_settings(NULL, NULL, NULL)
    together <- nnfor:::torch_train_standard(x, y, 4, 3, act, settings, rp, start = sw)
    for (r in 1:3) {
      alone <- nnfor:::torch_train_standard(x, y, 4, 1, act, settings, rp,
                                            start = sw[r, , drop = FALSE])
      expect_equal(alone$P[1, ], together$P[r, ], tolerance = 1e-5)
      expect_equal(alone$steps, together$steps[r])
    }
  }
})

test_that("neuralnet names are translated and torch names accepted", {
  skip_if_no_torch()
  expect_identical(nnfor:::torch_activation("logistic"), nnfor:::torch_activation("sigmoid"))
  expect_equal(nnfor:::resolve_torch_algorithm("rprop+"), "rprop")
  expect_equal(nnfor:::resolve_torch_algorithm("backprop"), "sgd")
  expect_error(nnfor:::resolve_torch_algorithm("lbfgs"), "couples the ensemble members")
  expect_error(nnfor:::torch_activation("nope"), "act.fct must be")
  set.seed(1)
  fit <- quiet(mlp(short_y(), reps = 2, backend = "torch", act.fct = "tanh", algorithm = "rprop+",
                   stepmax = 200))
  expect_equal(fit$net$settings$algorithm, "rprop")
  expect_warning(mlp(short_y(), reps = 1, backend = "torch", stepmax = 50, lifesign = "full"),
                 "not used by the torch backend: lifesign")
})

test_that("torch fits are reproducible with set.seed", {
  skip_if_no_torch()
  fit <- function() { set.seed(3); quiet(mlp(short_y(), reps = 3, backend = "torch", stepmax = 300)) }
  a <- fit(); b <- fit()
  expect_identical(a$net$P, b$net$P)
  expect_identical(forecast(a, h = 6)$mean, forecast(b, h = 6)$mean)
})

test_that("custom architectures fit, forecast, print and plot", {
  skip_if_no_torch()
  arch <- function(n_in) torch::nn_sequential(torch::nn_linear(n_in, 8), torch::nn_relu(),
                                              torch::nn_dropout(0.1), torch::nn_linear(8, 4),
                                              torch::nn_tanh())
  set.seed(2)
  fit <- quiet(mlp(short_y(), reps = 2, hd = arch, backend = "torch", algorithm = "adam",
                   learningrate = 0.01, stepmax = 200))
  expect_equal(fit$net$type, "custom")
  expect_length(forecast(fit, h = 6)$mean, 6)
  expect_output(print(fit), "custom architecture")
  pdf(NULL); on.exit(dev.off())
  expect_silent(plot(fit))
  set.seed(2)
  again <- quiet(mlp(short_y(), reps = 2, hd = arch, backend = "torch", algorithm = "adam",
                     learningrate = 0.01, stepmax = 200))
  expect_identical(fit$fitted, again$fitted)
  expect_error(mlp(short_y(), hd = arch), "needs backend")
  expect_warning(mlp(short_y(), reps = 1, hd = arch, backend = "torch", hd.auto.type = "cv",
                     stepmax = 20), "hd.auto.type is ignored")
})

test_that("torch fits survive saving and reloading", {
  skip_if_no_torch()
  set.seed(4)
  arch <- function(n_in) torch::nn_sequential(torch::nn_linear(n_in, 3), torch::nn_tanh())
  for (fit in list(quiet(mlp(short_y(), reps = 2, backend = "torch", stepmax = 100)),
                   quiet(mlp(short_y(), reps = 2, hd = arch, backend = "torch", stepmax = 100)))) {
    f <- tempfile(fileext = ".rds")
    saveRDS(fit, f)
    expect_equal(forecast(readRDS(f), h = 4)$mean, forecast(fit, h = 4)$mean, tolerance = 1e-6)
  }
})

test_that("hidden-node search and model reuse work with torch", {
  skip_if_no_torch()
  set.seed(1)
  fit <- quiet(mlp(short_y(), reps = 2, backend = "torch", hd.auto.type = "valid", hd.max = 2,
                   stepmax = 100))
  expect_true(fit$hd %in% 1:2)
  re <- quiet(mlp(AirPassengers, model = fit))
  expect_equal(re$backend, "torch")
  expect_identical(re$net$P, fit$net$P)
  set.seed(1)
  rt <- quiet(mlp(AirPassengers, model = fit, retrain = TRUE))
  expect_equal(rt$net$settings, fit$net$settings)
})

test_that("compare_backends reports every configuration", {
  skip_if_no_torch()
  cmp <- quiet(compare_backends(short_y(), reps = 2, configs = list(
    nn = list(backend = "neuralnet"), th = list(backend = "torch", stepmax = 100))))
  expect_equal(cmp$results$configuration, c("nn", "th"))
  expect_true(all(is.finite(cmp$results$MSE.out)))
  expect_output(print(cmp), "MSE out-of-sample")
  pdf(NULL); on.exit(dev.off())
  expect_silent(plot(cmp))
})
