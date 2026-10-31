# Construction of network inputs: lags of the target, lags of exogenous
# regressors and deterministic seasonal dummies.

# ---------------------------------------------------------------------------
# Lag specification
# ---------------------------------------------------------------------------

# Fill in default lags and keep vectors and validate them.
default_lags <- function(lags, keep, ff, xreg.lags, xreg.keep, xreg) {
  if (is.null(lags)) {
    lags <- if (max(ff) > 3) 1:max(ff) else 1:4
  }
  if (!is.null(xreg) && is.null(xreg.lags)) {
    xreg.lags <- rep(list(lags), ncol(xreg))
  }

  if (!is.null(keep)) {
    if (length(keep) != length(lags)) {
      stop("Argument `keep' must be a logical vector of length equal to lags.", call. = FALSE)
    }
  } else {
    keep <- rep(FALSE, length(lags))
  }
  if (!has_lags(lags)) {
    keep <- NULL
  }

  if (!is.null(xreg.keep)) {
    if (!all(lengths(xreg.lags) == lengths(xreg.keep))) {
      stop("Argument `xreg.keep' must be a list of logical vectors of length equal to the length of the lags in xreg.lags.",
           call. = FALSE)
    }
  } else {
    xreg.keep <- lapply(lengths(xreg.lags), function(n) rep(FALSE, n))
  }

  list(lags = lags, keep = keep, xreg.lags = xreg.lags, xreg.keep = xreg.keep)
}

# Largest lag across target and regressors; this many initial observations
# are consumed when building the training set.
max_lag <- function(lags, xreg.lags) {
  ylag <- if (has_lags(lags)) max(lags) else 0
  xlag <- vapply(xreg.lags, function(l) if (length(l)) max(l) else 0, numeric(1))
  max(c(ylag, xlag))
}

# ---------------------------------------------------------------------------
# Lagged inputs
# ---------------------------------------------------------------------------

# Lagged inputs for target positions `t` of the (differenced, scaled) series.
#
# This single definition is used both to build the training matrix (many
# positions) and, during recursive forecasting, a single input row whose
# history has been extended with earlier forecasts. Keeping one definition
# guarantees the two can never disagree on column order or alignment.
#
# Returns list(X = univariate lags, Xreg = list with one matrix (or NULL)
# per regressor). Column names are X<lag> and Xreg.<i>.<lag>.
lag_inputs <- function(y.sc, xreg.sc, lags, xreg.lags, t) {
  y.sc <- as.vector(y.sc)
  X <- NULL
  if (has_lags(lags)) {
    X <- matrix(vapply(lags, function(l) y.sc[t - l], numeric(length(t))),
                nrow = length(t), dimnames = list(NULL, paste0("X", lags)))
  }
  Xreg <- NULL
  if (!is.null(xreg.sc)) {
    Xreg <- lapply(seq_len(ncol(xreg.sc)), function(i) {
      l <- xreg.lags[[i]]
      if (!length(l)) return(NULL)
      matrix(vapply(l, function(k) xreg.sc[t - k, i], numeric(length(t))),
             nrow = length(t), dimnames = list(NULL, paste0("Xreg.", i, ".", l)))
    })
  }
  list(X = X, Xreg = Xreg)
}

# Training target and lagged inputs from the scaled, differenced series.
training_inputs <- function(y.sc, xreg.sc, lags, xreg.lags) {
  lag.max <- max_lag(lags, xreg.lags)
  t <- (lag.max + 1):length(y.sc)
  if (!is.null(xreg.sc) && nrow(xreg.sc) < length(y.sc)) {
    stop("Length of xreg after construction of lags smaller than training sample.", call. = FALSE)
  }
  inp <- lag_inputs(y.sc, xreg.sc, lags, xreg.lags, t)
  inp$Y <- matrix(as.vector(y.sc)[t], ncol = 1, dimnames = list(NULL, "Y"))
  inp$lag.max <- lag.max
  inp
}

# ---------------------------------------------------------------------------
# Deterministic seasonality
# ---------------------------------------------------------------------------

# Should seasonality be modelled with dummies? Not if a single seasonality is
# already removed by seasonal differencing, or if no seasonality was found.
use_season_dummies <- function(st, difforder, ff, allow.det.season) {
  seasonally.differenced <- length(ff) == 1 && any(difforder == max(ff))
  !seasonally.differenced && max(ff) > 1 && isTRUE(st$season.exist) && allow.det.season
}

resolve_det_type <- function(det.type, ff) {
  if (det.type != "auto") return(det.type)
  if (length(ff) == 1 && ff[1] <= 12) "bin" else "trg"
}

# One block of dummies for seasonal period `m`. Trigonometric dummies keep
# only the first harmonic (the full set caused problems for fractional
# periods < 3). `y` only supplies the seasonal phase via its start().
dummy_block <- function(n, m, y, type, id) {
  if (type == "trg") {
    D <- tsutils::seasdummy(n, m = m, y = y, type = "trg", full = TRUE)
    D <- D[, 1:min(ncol(D), 2), drop = FALSE]
  } else {
    D <- tsutils::seasdummy(n, m = m, y = y, type = "bin")
  }
  colnames(D) <- paste0("D", id, ".", seq_len(ncol(D)))
  D
}

# Seasonal dummies are anchored on the observation index of y, so the
# in-sample and forecast dummies always line up, whatever the periods.
# When a period equals the frequency of y, the season of the first
# observation is taken from start(y); otherwise the first observation is
# season 1 (e.g. when `m` overrides the frequency of y).
season_offset <- function(y, m) {
  st <- start(y)
  if (isTRUE(all.equal(m, frequency(y))) && length(st) == 2) st[2] - 1 else 0
}

# Dummies for `n` consecutive observations, the first being observation
# `first` of y (first > length(y) for forecasts). Returns one block per period.
season_dummies <- function(y, first, n, periods, type) {
  lapply(seq_along(periods), function(s) {
    m <- periods[s]
    k <- first + season_offset(y, m)
    ref <- ts(numeric(n), start = 1 + (k - 1) / m, frequency = m)  # start() gives the phase
    dummy_block(n, m, ref, type, s)
  })
}

# Dummies aligned with the last `n` in-sample observations of `y`.
season_dummies_fit <- function(n, periods, type, y) {
  season_dummies(y, length(y) - n + 1, n, periods, type)
}

# Dummies for the `h` periods following the sample.
season_dummies_forecast <- function(h, periods, type, y) {
  cbind_list(season_dummies(y, length(y) + 1, h, periods, type))
}

# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------

# Select inputs with backwards stepwise regression (AIC), falling back to
# lasso if the regression cannot be estimated (e.g. too few observations).
# Each seasonal dummy block enters as a single term, so binary dummies are
# kept or dropped as a group.
#
# Returns the retained lags, xreg lags and a logical per seasonal period
# telling whether its dummies remain.
select_inputs <- function(Y, X, Xreg, Xd, lags, keep, xreg.lags, xreg.keep, det.type) {
  x.n <- length(Xreg)
  Xreg.all <- cbind_list(Xreg)
  dat <- as.data.frame(cbind(Y, X, Xreg.all))
  if (is.null(Xd) && ncol(dat) == 1) {
    stop("Cannot build a network with no univariate or exogenous lags and no deterministic seasonality. Increase the maximum lags.",
         call. = FALSE)
  }
  # Dummy blocks are stored as matrix columns so that each is one model term.
  season.terms <- paste0("Xd.", seq_along(Xd))
  for (i in seq_along(Xd)) dat[[season.terms[i]]] <- Xd[[i]]

  keep.all <- c(keep, unlist(xreg.keep), rep(FALSE, length(Xd)))
  fit <- lm(Y ~ ., data = dat)

  cf <- tryCatch({
    scp <- NULL
    if (any(keep.all)) {
      forced <- attr(fit$terms, "term.labels")[keep.all]
      scp <- list(lower = as.formula(paste("~", paste(forced, collapse = " + "))))
    }
    coef(MASS::stepAIC(fit, trace = 0, direction = "backward", scope = scp))
  }, error = function(e) {
    if (any(keep.all)) {
      warning("Cannot execute backwards variable selection, reverting to Lasso. Arguments `keep' and `xreg.keep' will be ignored.")
    }
    lasso.X <- cbind(X, Xreg.all)
    for (i in seq_along(Xd)) {
      D <- Xd[[i]]
      colnames(D) <- paste0(season.terms[i], colnames(D))
      lasso.X <- cbind(lasso.X, D)
    }
    fit.lasso <- suppressWarnings(glmnet::cv.glmnet(x = lasso.X, y = Y[, 1]))
    cf <- as.vector(coef(fit.lasso))
    names(cf) <- rownames(coef(fit.lasso))
    cf[cf != 0]
  })
  kept <- names(cf)

  lags.new <- lags[colnames(X) %in% kept]
  if (x.n > 0) {
    xreg.lags <- lapply(seq_len(x.n), function(i) {
      xreg.lags[[i]][colnames(Xreg[[i]]) %in% kept]
    })
  }

  # Trigonometric dummies are rarely retained by a linear regression, so
  # only binary dummies may be removed by the selection.
  still.det <- rep(TRUE, length(Xd))
  if (det.type == "bin") {
    still.det <- vapply(seq_along(Xd), function(i) {
      any(paste0(season.terms[i], colnames(Xd[[i]])) %in% kept)
    }, logical(1))
  }

  # The regression may reject every input, in which case force an AR(1).
  if (length(lags.new) + sum(lengths(xreg.lags)) + sum(still.det) == 0) {
    warning("No inputs left in the network after pre-selection, forcing AR(1).")
    lags.new <- 1
  }
  # If all univariate lags are rejected but regressors or dummies remain, the
  # network uses those only (lags = 0 means no univariate lags).
  lags <- if (length(lags.new) > 0) lags.new else 0

  list(lags = lags, xreg.lags = xreg.lags, still.det = still.det)
}
