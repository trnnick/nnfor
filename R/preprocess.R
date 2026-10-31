# Preprocessing pipeline shared by mlp() and elm().
#
# `ff` holds the seasonal periods to model (the `m` argument of mlp/elm, or
# the periods of the ts/msts object).
#
# Turns a time series (plus optional regressors) into a training set for a
# network and records everything needed to repeat the transformation when
# forecasting. Model-agnostic: the output can feed any backend.
#
# Steps:
#   1. seasonality / trend analysis (skipped when a model is reused)
#   2. differencing and scaling of the target
#   3. scaling of regressors
#   4. lagged inputs and seasonal dummies
#   5. optional input selection, after which inputs are rebuilt
#
# `fixed` is used when re-fitting an existing model to new data: it carries
# the model's seasonal specification so that the exact same inputs are built
# instead of re-detecting them from the new data.

preprocess <- function(y, ff, lags, keep, difforder, sel.lag, allow.det.season, det.type,
                       xreg, xreg.lags, xreg.keep, fixed = NULL) {

  # 1. Seasonality and differencing
  if (is.null(fixed)) {
    # When m overrides the frequency of y, analyse the series by position
    # rather than by the calendar of the ts object.
    y.an <- if (isTRUE(all.equal(max(ff), frequency(y)))) y else as.numeric(y)
    cma <- tsutils::cmav(y.an, ma = max(ff))
    st <- seasoncheck(y.an, m = max(ff), cma = cma)
    difforder <- resolve_difforder(difforder, y.an, ff, st)
    sdummy <- use_season_dummies(st, difforder, ff, allow.det.season)
    if (sdummy) det.type <- resolve_det_type(det.type, ff)
    ff.det <- if (sdummy) ff else NULL
  } else {
    difforder <- resolve_difforder(difforder, y, ff, st = NULL)
    sdummy <- fixed$sdummy
    ff.det <- fixed$ff.det
    det.type <- fixed$det.type
  }

  # 2. Target: difference then scale
  y.d <- difference_series(y, difforder)
  sc <- linscale(y.d[[length(y.d)]], minmax = nnfor_scale_range())
  y.sc <- sc$x

  # 3. Regressors: align with the differenced target and scale
  xsc <- scale_xreg(xreg, sum(difforder))

  # 4. Inputs
  inp <- training_inputs(y.sc, xsc$x, lags, xreg.lags)
  Xd <- if (sdummy) season_dummies_fit(nrow(inp$Y), ff.det, det.type, y) else NULL

  # 5. Input selection
  if (sel.lag) {
    sel <- select_inputs(inp$Y, inp$X, inp$Xreg, Xd, lags, keep,
                         xreg.lags, xreg.keep, det.type)
    lags <- sel$lags
    xreg.lags <- sel$xreg.lags
    inp <- training_inputs(y.sc, xsc$x, lags, xreg.lags)
    if (sdummy) {
      ff.det <- ff.det[sel$still.det]
      sdummy <- length(ff.det) > 0
      Xd <- if (sdummy) season_dummies_fit(nrow(inp$Y), ff.det, det.type, y) else NULL
      if (!sdummy) ff.det <- NULL
    }
  }

  X <- cbind(inp$X, cbind_list(inp$Xreg), cbind_list(Xd))
  if (is.null(X) || ncol(X) == 0) {
    stop("Cannot build a network with no univariate or exogenous lags and no deterministic seasonality.",
         call. = FALSE)
  }

  list(Y = inp$Y, X = X, lag.max = inp$lag.max,
       lags = lags, xreg.lags = xreg.lags, difforder = difforder,
       ff = ff, sdummy = sdummy, ff.det = ff.det, det.type = det.type,
       minmax = sc$minmax, xreg.minmax = xsc$minmax, y.d = y.d)
}
