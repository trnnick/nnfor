test_that("elm barebone accepts type = 'lm'", {
  set.seed(1)
  expect_s3_class(quiet(elm(AirPassengers, reps = 1, barebone = TRUE, type = "lm")), "elm.fast")
})

test_that("multi-layer ELM fits and forecasts", {
  set.seed(1)
  fit <- quiet(elm(AirPassengers, reps = 1, hd = c(6, 4)))
  expect_length(forecast(fit, h = 6)$mean, 6)
  expect_error(elm(AirPassengers, reps = 1, hd = c(6, 4), barebone = TRUE), "single hidden layer")
})

test_that("one-step forecast works with trigonometric dummies", {
  set.seed(1)
  fit <- quiet(mlp(AirPassengers, reps = 1, det.type = "trg", difforder = 1))
  expect_length(forecast(fit, h = 1)$mean, 1)
})

test_that("multiple seasonalities with binary dummies fit and forecast", {
  set.seed(1)
  y <- forecast::msts(200 + 10 * sin(2 * pi * (1:240) / 4) + 20 * sin(2 * pi * (1:240) / 12) +
                        rnorm(240), seasonal.periods = c(4, 12))
  fit <- quiet(mlp(y, reps = 1, difforder = 0, det.type = "bin"))
  expect_length(forecast(fit, h = 12)$mean, 12)
})

test_that("thief wrappers pass xreg to forecast correctly", {
  set.seed(1)
  X <- cbind(rnorm(156))
  out <- quiet(mlp.thief(AirPassengers, h = 12, reps = 1, xreg = X))
  expect_length(out$mean, 12)
  expect_length(out$fitted, length(AirPassengers))
})

test_that("forecasting without required xreg gives a clear error", {
  set.seed(1)
  X <- cbind(rnorm(156))
  fit <- quiet(mlp(AirPassengers, reps = 1, xreg = X, xreg.lags = list(0), sel.lag = FALSE))
  expect_error(forecast(fit, h = 12), "provide xreg")
})

test_that("objects from earlier versions still forecast, print and plot", {
  set.seed(1)
  fit <- quiet(mlp(AirPassengers, reps = 2))
  old <- unclass(fit)[setdiff(names(fit), c("fitted.all", "backend", "ff"))]
  class(old) <- "mlp"
  expect_equal(forecast(old, h = 6)$mean, forecast(fit, h = 6)$mean)
  expect_output(print(old), "MLP fit")
  pdf(NULL); on.exit(dev.off())
  expect_silent(plot(old, which = "fit"))
})
