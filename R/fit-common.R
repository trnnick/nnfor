# Scaffolding shared by mlp() and elm(): everything before and after the
# network-specific training step.

# Check that a previously fitted model can be applied to new data.
check_reuse_model <- function(model, cls, xreg, ff) {
  if (!inherits(model, cls)) {
    stop("model must be an ", cls, " object, the output of the ", cls, "() function.", call. = FALSE)
  }
  if (n_xreg(xreg) != length(model$xreg.minmax)) {
    stop("Previous model xreg specification and new xreg inputs do not match.", call. = FALSE)
  }
  if (isTRUE(model$sdummy) && !all(model$ff.det %in% ff)) {
    stop("Seasonality of current data and seasonality of provided model does not match.", call. = FALSE)
  }
  invisible(model)
}

# Input specification inherited from a reused model. The model's structure is
# copied exactly (no re-detection), so the inputs match the trained network.
reused_spec <- function(model) {
  list(lags = model$lags, keep = NULL, xreg.lags = model$xreg.lags, xreg.keep = NULL,
       difforder = model$difforder %||% 0,  # NULL means "none" in a fitted model
       comb = model$comb, sel.lag = FALSE,
       fixed = list(sdummy = model$sdummy, ff.det = model$ff.det, det.type = model$det.type))
}

# Run preprocessing for a new or reused model.
prepare_data <- function(y, ff, lags, keep, difforder, sel.lag, allow.det.season, det.type,
                         xreg, xreg.lags, xreg.keep, model = NULL) {
  if (!is.null(model)) {
    spec <- reused_spec(model)
    lags <- spec$lags
    xreg.lags <- spec$xreg.lags
    difforder <- spec$difforder
    sel.lag <- FALSE
  }
  dl <- default_lags(lags, if (is.null(model)) keep, ff, xreg.lags,
                     if (is.null(model)) xreg.keep, xreg)
  preprocess(y, ff, dl$lags, dl$keep, difforder, sel.lag, allow.det.season, det.type,
             xreg, dl$xreg.lags, dl$xreg.keep,
             fixed = if (!is.null(model)) reused_spec(model)$fixed)
}

# Member predictions for all rows of X as an n x reps matrix.
predict_members <- function(predict_fun, X, reps) {
  matrix(vapply(seq_len(reps), function(r) as.vector(predict_fun(X, r)), numeric(nrow(X))),
         nrow = nrow(X))
}

# Turn scaled, differenced member predictions into fitted values on the
# original scale, combine them and compute the in-sample MSE.
finalise_fit <- function(Yhat.sc, pp, y, comb) {
  Yhat <- apply(Yhat.sc, 2, function(v) {
    undifference_fitted(linscale(v, pp$minmax, rev = TRUE)$x, pp$y.d, pp$difforder)
  })
  Yhat <- matrix(Yhat, ncol = ncol(Yhat.sc))
  fitted <- ts(frc.comb(Yhat, comb), end = end(y), frequency = frequency(y))
  fitted.all <- ts(Yhat, end = end(y), frequency = frequency(y))
  if (ncol(Yhat) > 1) colnames(fitted.all) <- paste0("NN.", seq_len(ncol(Yhat)))
  first <- pp$lag.max + 1 + sum(pp$difforder)
  list(fitted = fitted, fitted.all = fitted.all,
       MSE = mean((y[first:length(y)] - fitted)^2))
}
