# Results must not depend on the number of cores. Workers use no random
# numbers: everything random is drawn before members are dispatched.
with_cores <- function(cores, expr) {
  old <- options(nnfor.cores = cores)
  on.exit(options(old))
  set.seed(42)
  quiet(expr)
}

test_that("parallel and sequential fits are identical", {
  skip_on_cran()
  y <- window(AirPassengers, end = c(1955, 12))
  for (fit_fun in list(function() mlp(y, reps = 3, hd = 3),
                       function() elm(y, reps = 3),
                       function() elm(y, reps = 3, barebone = TRUE, direct = TRUE))) {
    a <- with_cores(1, fit_fun())
    b <- with_cores(2, fit_fun())
    expect_identical(a$fitted.all, b$fitted.all)
    expect_identical(forecast(a, h = 6)$all.mean, forecast(b, h = 6)$all.mean)
  }
  X <- matrix(rnorm(200), 50); Y <- X %*% c(1, -1, 0.5, 2)
  expect_identical(with_cores(1, elm.fast(Y, X, reps = 3, hd = 8))$fitted,
                   with_cores(2, elm.fast(Y, X, reps = 3, hd = 8))$fitted)
})

test_that("starting weights are drawn as neuralnet draws them", {
  dat <- data.frame(Y = rnorm(20), X1 = rnorm(20), X2 = rnorm(20))
  for (hd in list(3, c(4, 2))) {
    set.seed(1)
    net <- neuralnet::neuralnet(Y ~ X1 + X2, dat, hidden = hd, threshold = 1e10)
    expect_length(unlist(net$startweights), nnfor:::neuralnet_n_weights(2, hd))
  }
})

test_that("members that fail to converge are dropped", {
  set.seed(9)
  expect_warning(fit <- mlp(AirPassengers, reps = 6, hd = 8, stepmax = 1000, sel.lag = FALSE),
                 "did not converge in 4 of 6")
  expect_equal(nnfor:::n_members(fit), 2)
  expect_equal(ncol(fit$fitted.all), 2)
  expect_length(forecast(fit, h = 3)$mean, 3)
})

test_that("user-supplied starting weights are passed to neuralnet", {
  set.seed(1)
  fit <- quiet(mlp(AirPassengers, reps = 2, hd = 2, lags = 1:2, difforder = 1,
                   allow.det.season = FALSE, sel.lag = FALSE, startweights = rep(0.1, 9)))
  expect_equal(unname(unlist(fit$net$startweights[[1]])), rep(0.1, 9))
})

test_that("invalid nnfor.cores is rejected", {
  old <- options(nnfor.cores = 0); on.exit(options(old))
  expect_error(mlp(AirPassengers, reps = 1), "nnfor.cores")
})
