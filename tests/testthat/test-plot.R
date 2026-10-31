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
