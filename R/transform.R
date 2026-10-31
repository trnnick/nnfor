# Transformations applied to the target series before training, and their
# inverses. Every model shares these, so they live in one place.

# Decide the differencing order.
#   NULL -> identify automatically (trend and seasonal unit-root checks)
#   0    -> no differencing
# Returns NULL when no differencing is to be applied.
resolve_difforder <- function(difforder, y, ff, st) {
  if (any(difforder == 0)) return(NULL)
  if (!is.null(difforder)) return(difforder)

  difforder <- 0
  if (trendcheck(st$cma)) {
    difforder <- 1
  }
  if (max(ff) > 1 && isTRUE(st$season.exist)) {
    # Remove the trend appropriately before testing for a seasonal unit root.
    # The multiplicative test needs at least three full seasons.
    m.seas <- if (length(y) / max(ff) < 3) {
      TRUE
    } else {
      tsutils::mseastest(y, m = max(ff), cma = st$cma)$is.multiplicative
    }
    y.dt <- if (m.seas) y / st$cma else y - st$cma
    if (forecast::nsdiffs(ts(y.dt, frequency = max(ff)), test = "ch") > 0) {
      difforder <- c(difforder, max(ff))
    }
  }
  if (any(difforder == 0)) NULL else difforder
}

# Apply successive differences. Returns a list holding the series at every
# stage (d0, d0d1, d0d1d12, ...); all stages are needed to undifference.
difference_series <- function(y, difforder) {
  y.d <- list(d0 = y)
  for (k in difforder) {
    prev <- y.d[[length(y.d)]]
    y.d[[paste0(names(y.d)[length(y.d)], "d", k)]] <- diff(prev, k)
  }
  y.d
}

# Undifference in-sample fitted values. These are one-step-ahead fits, so
# every level is reconstructed from the *observed* values of the level above.
undifference_fitted <- function(yhat, y.d, difforder) {
  out <- as.vector(yhat)
  for (lv in rev(seq_along(difforder))) {
    k <- difforder[lv]
    obs <- y.d[[lv]]
    n.d <- length(obs)
    out <- obs[(n.d - length(out) - k + 1):(n.d - k)] + out
  }
  out
}

# Undifference a multi-step forecast. Here each level is reconstructed
# recursively from the tail of the observed series and the forecasts.
undifference_forecast <- function(frc, y.d, difforder) {
  out <- frc
  for (lv in rev(seq_along(difforder))) {
    k <- difforder[lv]
    tmp <- c(tail(y.d[[lv]], k), out)
    for (j in seq_len(length(tmp) - k)) {
      tmp[k + j] <- tmp[j] + tmp[k + j]
    }
    out <- tmp[(k + 1):length(tmp)]
  }
  out
}

# Scale each xreg column and drop the rows lost to differencing of y.
# Returns the scaled matrix and the per-column scaling information.
scale_xreg <- function(xreg, n.drop) {
  if (is.null(xreg)) return(list(x = NULL, minmax = NULL))
  xreg <- xreg[(n.drop + 1):nrow(xreg), , drop = FALSE]
  sc <- lapply(seq_len(ncol(xreg)), function(i) {
    linscale(xreg[, i], minmax = nnfor_scale_range())
  })
  x <- do.call(cbind, lapply(sc, `[[`, "x"))
  list(x = x, minmax = lapply(sc, `[[`, "minmax"))
}

# Apply stored scaling to new xreg values (used when forecasting).
rescale_xreg <- function(xreg, minmax, n.drop) {
  for (i in seq_len(ncol(xreg))) {
    xreg[, i] <- linscale(xreg[, i], minmax = minmax[[i]])$x
  }
  xreg[(n.drop + 1):nrow(xreg), , drop = FALSE]
}
