test_that("training and forecasting build identical input rows", {
  y.sc <- as.numeric(1:30)
  xreg <- cbind(101:140, 201:240)
  lags <- c(1, 3); xl <- list(0:1, 2)
  tr <- nnfor:::training_inputs(y.sc, xreg, lags, xl)
  # The training row for position t equals a forecast-style row built at t.
  t <- tr$lag.max + 5
  one <- nnfor:::lag_inputs(y.sc, xreg, lags, xl, t)
  expect_equal(tr$X[5, ], one$X[1, ])
  expect_equal(do.call(cbind, tr$Xreg)[5, ], do.call(cbind, one$Xreg)[1, ])
  expect_equal(colnames(tr$X), c("X1", "X3"))
  expect_equal(unlist(lapply(tr$Xreg, colnames)), c("Xreg.1.0", "Xreg.1.1", "Xreg.2.2"))
})

test_that("forecast dummies continue the in-sample seasonal pattern", {
  y <- ts(rnorm(30), start = c(2000, 3), frequency = 4)
  for (type in c("bin", "trg")) {
    fit <- nnfor:::season_dummies_fit(30, 4, type, y)[[1]]
    frc <- nnfor:::season_dummies_forecast(8, 4, type, y)
    expect_equal(frc[1:4, ], fit[27:30, ], ignore_attr = TRUE)
    expect_equal(frc[5:8, ], frc[1:4, ], ignore_attr = TRUE)
  }
})

test_that("dummies follow start(y) when m is the series frequency", {
  y <- ts(rnorm(8), start = c(2000, 3), frequency = 4)
  D <- nnfor:::season_dummies_fit(8, 4, "bin", y)[[1]]
  expect_equal(unname(D[1, ]), c(0, 0, 1))   # first observation is Q3
})

test_that("binary dummies stay in phase to the end of the sample", {
  # tsutils::seasdummy() < 0.9.5 mislabelled the last observations when the
  # sample did not start at season 1 and n was not a multiple of m.
  y <- ts(rnorm(10), start = c(2000, 3), frequency = 4)
  D <- nnfor:::season_dummies_fit(10, 4, "bin", y)[[1]]
  season <- apply(cbind(D, 1 - rowSums(D)), 1, which.max)
  expect_equal(season, ((2 + 0:9) %% 4) + 1)
})

test_that("fractional-period dummies continue smoothly into the forecast", {
  y <- ts(rnorm(40), frequency = 1)
  fit <- nnfor:::season_dummies_fit(40, 7.5, "trg", y)[[1]]
  frc <- nnfor:::season_dummies_forecast(5, 7.5, "trg", y)
  t <- 1:45
  truth <- cbind(cos(2 * pi * t / 7.5), sin(2 * pi * t / 7.5))
  expect_equal(unname(rbind(fit, frc)), truth)
})
