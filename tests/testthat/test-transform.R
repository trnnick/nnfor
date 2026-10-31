test_that("linscale round-trips and reuses a stored range", {
  x <- c(5, 1, 9, 3)
  sc <- linscale(x, minmax = list(mn = -0.8, mx = 0.8))
  expect_equal(range(sc$x), c(-0.8, 0.8))
  expect_equal(linscale(sc$x, sc$minmax, rev = TRUE)$x, x)
  expect_equal(linscale(c(1, 9), minmax = sc$minmax)$x, c(-0.8, 0.8))
  expect_error(linscale(x, rev = TRUE), "minmax input is required")
})

test_that("in-sample undifferencing inverts differencing", {
  y <- ts(cumsum(cumsum(rnorm(40))) + 100, frequency = 4)
  d <- c(1, 4)
  y.d <- nnfor:::difference_series(y, d)
  last <- y.d[[length(y.d)]]
  expect_equal(nnfor:::undifference_fitted(last, y.d, d), as.numeric(y)[(sum(d) + 1):40])
})

test_that("forecast undifferencing continues the observed series", {
  y <- ts(1:20 + rep(c(0, 5, 2, 7), 5), frequency = 4)
  y.d <- nnfor:::difference_series(y, c(1, 4))
  # A zero forecast of the doubly differenced series continues trend and season.
  out <- nnfor:::undifference_forecast(rep(0, 4), y.d, c(1, 4))
  expect_equal(out, 21:24 + c(0, 5, 2, 7))
})
