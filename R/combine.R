# Combination of ensemble members.
#
# For a discussion of mean, median and mode combination see:
# Kourentzes, N., Barrow, D. K., & Crone, S. F. (2014). Neural network
# ensemble operators for time series forecasting. Expert Systems with
# Applications, 41(9), 4235-4244.

# Combine an n x reps matrix of member predictions into one vector.
frc.comb <- function(Yhat, comb, na.rm = FALSE) {
  na.rm <- isTRUE(na.rm[1])
  if (ncol(Yhat) == 1) return(Yhat[, 1])
  switch(comb,
         "median" = apply(Yhat, 1, median, na.rm = na.rm),
         "mean"   = apply(Yhat, 1, mean, na.rm = na.rm),
         "mode"   = comb_mode(Yhat),
         stop("Unknown combination operator: ", comb, call. = FALSE))
}

# Mode via kernel density estimation, row by row. Members containing NAs are
# dropped, and rows where all members agree are returned as is.
comb_mode <- function(Yhat) {
  Yok <- Yhat[, colSums(is.na(Yhat)) == 0, drop = FALSE]
  out <- Yhat[, 1]
  varying <- !apply(Yok, 1, forecast::is.constant)
  if (any(varying)) {
    out[varying] <- apply(Yok[varying, , drop = FALSE], 1, function(v) kdemode(v)$mode)
  }
  out
}

# Mode of a vector estimated with KDE.
#   type: bandwidth selection, "SJ" (Sheather & Jones) or "nrd0" (Silverman).
kdemode <- function(data, type = c("SJ", "nrd0")) {
  type <- match.arg(type)
  pad <- 0.1 * diff(range(data))
  ks <- density(data, bw = type, n = 512, from = min(data) - pad, to = max(data) + pad)
  list(mode = ks$x[which.max(ks$y)], xd = ks$x, fd = ks$y, h = ks$bw)
}
