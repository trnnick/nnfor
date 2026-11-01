test_that("plots draw without changing the user's graphical settings", {
  set.seed(1)
  fm <- quiet(mlp(AirPassengers, reps = 2))
  fe <- quiet(elm(AirPassengers, reps = 2, direct = TRUE))
  ff <- quiet(elm(AirPassengers, reps = 2, barebone = TRUE))
  fr <- forecast(fm, h = 6)
  pdf(NULL); on.exit(dev.off())
  par(mfrow = c(2, 2), fg = "red", lwd = 2)
  before <- user_settings()
  plot(fm); plot(fe); plot(ff)
  plot(fm, which = "fit"); plot(fr); plot(forecast(fe, h = 1))
  plot(elm.fast(rnorm(50), matrix(rnorm(100), 50), reps = 2, hd = 5))
  expect_identical(user_settings(), before)
})

test_that("plot customisation arguments are honoured", {
  set.seed(1)
  fm <- quiet(mlp(AirPassengers, reps = 2))
  fr <- forecast(fm, h = 6)
  pdf(NULL); on.exit(dev.off())
  expect_silent(plot(fm, main = "x", col.neuron = "gold", axis.args = NULL))
  expect_silent(plot(fm, which = "fit", members = FALSE, fitted.args = list(col = "red"), ylim = c(0, 700)))
  expect_silent(plot(fr, fitted = TRUE, members.args = list(lty = 2), mean.args = list(col = "red"),
                     xlim = c(1958, 1962)))
  plot(AirPassengers)
  expect_silent(plot(fr, add = TRUE))
  # The ensemble fit is drawn on the coordinates of the user's existing plot.
  plot(1:10, xlim = c(1949, 1962), ylim = c(0, 1000))
  usr <- par("usr")
  plot_fit(fm, add = TRUE)
  expect_identical(par("usr"), usr)
})

test_that("visible y-range follows xlim", {
  yl <- nnfor:::visible_ylim(list(AirPassengers), c(1949, 1950))
  expect_lt(yl[2], 200)
})

test_that("restrict limits forecast plots to twice the range of actuals and ensemble", {
  set.seed(1)
  fit <- quiet(mlp(AirPassengers, reps = 3))
  frc <- forecast(fit, h = 12)
  used <- function(...) {
    plot(...)
    u <- par("usr")[3:4]
    d <- diff(u) / 1.08
    c(u[1] + 0.04 * d, u[2] - 0.04 * d)
  }
  pdf(NULL); on.exit(dev.off())
  # Plots within the limit are unchanged
  expect_equal(used(frc), used(frc, restrict = FALSE))
  # A poorly trained member is cut off at twice the range
  frc$all.mean[, 2] <- frc$all.mean[, 2] * 6
  base <- range(frc$x, frc$mean)
  w <- diff(base)
  expect_equal(used(frc), c(base[1], base[2] + w / 2))
  expect_gt(used(frc, restrict = FALSE)[2], max(frc$all.mean) - 1)
  # A ylim given by the user wins
  expect_equal(used(frc, ylim = c(0, 5000)), c(0, 5000))
})
