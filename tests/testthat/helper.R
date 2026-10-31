# Small, fast fits shared by the tests.
quiet <- function(expr) suppressWarnings(expr)

seasonal_xreg_data <- function() {
  set.seed(6)
  n <- 96
  x <- rnorm(n + 12)
  y <- ts(100 + rep(c(10, -5, 3, -8, 6, 0, -2, 4, -6, 9, -3, -8), 8) + 8 * x[1:n] +
            rnorm(n, sd = 0.5), frequency = 12)
  list(y = y, x = cbind(x))
}

# Graphical settings a user may have chosen.
user_settings <- function() {
  p <- par(no.readonly = TRUE)
  # usr/xaxp/yaxp describe the last plot drawn; mfg/fig the current panel of
  # the user's layout, which advances with every plot by design.
  p[setdiff(names(p), c("usr", "xaxp", "yaxp", "mfg", "fig"))]
}
