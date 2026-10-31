test_that("rejected univariate lags are dropped when other inputs survive", {
  d <- seasonal_xreg_data()
  set.seed(42)
  fit <- quiet(mlp(d$y, reps = 1, xreg = d$x, xreg.lags = list(0:2), difforder = 0))
  expect_equal(fit$lags, 0)
  expect_true(fit$sdummy)
  expect_true(length(fit$xreg.lags[[1]]) > 0)
  expect_false(any(grepl("^X[0-9]", fit$net$model.list$variables)))
  expect_length(forecast(fit, h = 12, xreg = d$x)$mean, 12)
})

test_that("logistic elm.fast predictions do not depend on the other rows", {
  set.seed(1)
  X <- matrix(rnorm(400), 100)
  Y <- as.numeric(X %*% c(1, -1, 0.5, 0) > 0)
  fit <- elm.fast(Y, X, reps = 2, hd = 10, output = "logistic")
  batch <- predict(fit, X[1:5, ])$Y.hat
  single <- vapply(1:5, function(i) predict(fit, X[i, , drop = FALSE])$Y.hat, numeric(1))
  expect_equal(single, batch)
  expect_true(all(batch >= 0 & batch <= 1))
})

test_that("m overrides the seasonality of y", {
  y1 <- ts(as.numeric(AirPassengers))
  set.seed(42); a <- quiet(mlp(AirPassengers, reps = 1))
  set.seed(42); b <- quiet(mlp(y1, m = 12, reps = 1))
  expect_equal(as.numeric(b$fitted), as.numeric(a$fitted))
  expect_equal(b$ff, 12)
  expect_length(forecast(b)$mean, 12)

  set.seed(42); f1 <- quiet(mlp(AirPassengers, m = 1, reps = 1))
  expect_false(f1$sdummy)
  expect_equal(f1$ff, 1)
  expect_error(mlp(AirPassengers, m = 0), "seasonal periods")
})

test_that("a reused model keeps its specification and seasonal periods", {
  y1 <- ts(as.numeric(AirPassengers))
  set.seed(42)
  fit <- quiet(mlp(window(y1, end = 120), m = 12, reps = 1, difforder = 0))
  re <- quiet(mlp(y1, model = fit))
  expect_equal(re$ff, 12)
  expect_identical(re[c("lags", "difforder", "sdummy", "ff.det")],
                   fit[c("lags", "difforder", "sdummy", "ff.det")])
})
