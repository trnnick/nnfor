#' Extreme learning machines for time series forecasting
#'
#' This function fits ELM neural networks for time series forecasting.
#'
#' @param y Input time series. Can be ts or msts object.
#' @param m Seasonal period(s) of the series. Use a vector for multiple seasonalities, e.g. \code{c(7, 365.25)}. If NULL (default) they are taken from \code{y}: its frequency for a ts object, or all seasonal periods for an msts object. When given, \code{m} overrides the periods of \code{y} for lag specification, seasonality and differencing tests and seasonal dummies.
#' @param hd Number of hidden nodes. This can be a vector, where each number represents the number of hidden nodes of a different hidden layer. Use NULL to automatically specify.
#' @param type Estimation type for output layer weights. Can be "lasso" (lasso with CV), "ridge" (ridge regression with CV), "step" (stepwise regression with AIC) or "lm" (linear regression; "ls" is accepted as a synonym).
#' @param reps Number of networks to train, the result is the ensemble forecast.
#' @param comb Combination operator for forecasts when reps > 1. Can be "median", "mode" (based on KDE estimation) and "mean".
#' @param lags Lags of y to use as inputs. If none provided then 1:frequency(y) is used. Use 0 for no univariate lags.
#' @param keep Logical vector to force lags to stay in the model if sel.lag == TRUE. If NULL then it keep = rep(FALSE,length(lags)).
#' @param difforder Vector including the differencing lags. For example c(1,12) will apply first and seasonal (12) differences. For no differencing use 0. For automatic selection use NULL.
#' @param outplot Provide plot of model fit. Can be TRUE or FALSE. For a customised plot use \code{plot(fit, which = "fit", ...)} after fitting.
#' @param sel.lag Automatically select lags. Can be TRUE or FALSE.
#' @param direct Use direct input-output connections to model strictly linear effects. Can be TRUE or FALSE.
#' @param allow.det.season Permit modelling seasonality with deterministic dummies.
#' @param det.type Type of deterministic seasonality dummies to use. This can be "bin" for binary or "trg" for a sine-cosine pair. With "auto" if ony a single seasonality is used and periodicity is up to 12 then "bin" is used, otherwise "trg".
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set, but can be longer.
#' @param xreg.lags This is a list containing the lags for each exogenous variable. Each list is a numeric vector containing lags. If xreg has 3 columns then the xreg.lags list must contain three elements. If NULL then it is automatically specified.
#' @param xreg.keep List of logical vectors to force lags of xreg to stay in the model if sel.lag == TRUE. If NULL then all exogenous lags can be removed.
#' @param barebone Use an alternative elm implementation (written in R) that is faster when the number of inputs is very high. Typically not needed. Supports a single hidden layer only.
#' @param model A previously trained elm object. If this is provided then the same model is fitted to y, without re-estimating any model parameters.
#' @param retrain If a previous model is provided, retrain the network or not. If the network is retrained the size of the hidden layer is reset.
#'
#' @return Return object of class \code{elm}. If barebone == TRUE then the object inherits a second class "\code{elm.fast}".
#'   The function \code{plot} produces a plot the network architecture or of the fit (see \code{\link{plot.elm}}).
#'   \code{elm} contains:
#' \itemize{
#' \item \code{net}: ELM networks. If it is of class "\code{elm.fast}" then this is NULL.
#' \item \code{hd}: Number of hidden nodes. If it is of class "\code{elm.fast}" this is a vector with a different number for each repetition.
#' \item \code{W.in}: NULL unless it is of class "\code{elm.fast}". Contains the input weights.
#' \item \code{W}: Output layer weights for each repetition.
#' \item \code{b}: Contains the output node bias for each training repetition.
#' \item \code{W.dct}: Contains the direct connection weights if argument direct == TRUE. Otherwise is NULL.
#' \item \code{lags}: Input lags used.
#' \item \code{xreg.lags}: \code{xreg} lags used.
#' \item \code{difforder}: Differencing used.
#' \item \code{sdummy}: Use of deterministic seasonality.
#' \item \code{ff}: Seasonal periods modelled (\code{m}, or those of \code{y}).
#' \item \code{ff.det}: Seasonal frequencies coded using deterministic dummies.
#' \item \code{det.type}: Type of determistic seasonality.
#' \item \code{y}: Input time series.
#' \item \code{minmax}: Scaling structure.
#' \item \code{xreg.minmax}: Scaling structure for xreg variables.
#' \item \code{comb}: Combination operator used.
#' \item \code{type}: Estimation used for output layer weights.
#' \item \code{direct}: Presence of direct input-output connections.
#' \item \code{fitted}: Fitted values.
#' \item \code{MSE}: In-sample Mean Squared Error.
#' \item \code{fitted.all}: Fitted values of each ensemble member.
#' }
#'
#' @section Parallel training:
#' Ensemble members can be trained in parallel with
#' \code{options(nnfor.cores = 4)} (forked workers on Linux and macOS) or
#' \code{options(nnfor.cores = cl)} with a cluster from
#' \code{parallel::makeCluster()}, which is recommended on Windows. All random
#' numbers are drawn before members are dispatched, so results are identical
#' for any number of cores and reproducible with \code{set.seed()}.
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{forecast.elm}}, \code{\link{elm.thief}}, \code{\link{mlp}}.
#' @references
#' \itemize{
#' \item For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{https://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.
#' \item For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.
#' \item For variable selection see: Crone S.F., Kourentzes N. (2010) \href{https://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction -- A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.
#' \item For ELMs see: Huang G.B., Zhou H., Ding X. (2006) Extreme learning machine: theory and applications. \emph{Neurocomputing}, \bold{70}(\bold{1}), 489-501.
#' }
#' @keywords mlp thief ts
#' @note To use elm with Temporal Hierarchies (\href{https://cran.r-project.org/package=thief}{thief} package) see \code{\link{elm.thief}}.
#'   The elm function by default calls the \code{neuralnet} function. If barebone == TRUE then it uses an alternative implementation (\code{\link{elm.fast}}) which is more appropriate when the number of inputs is several hundreds.
#' @examples
#' \dontshow{
#'  fit <- elm(AirPassengers,reps=1)
#'  print(fit)
#'  plot(fit)
#' }
#' \dontrun{
#'  fit <- elm(AirPassengers)
#'  print(fit)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#' \dontrun{
#' # Data for the examples with exogenous regressors (xreg): a promotion
#' # indicator and a slowly moving price. xreg must cover the sample and the
#' # forecast horizon.
#' set.seed(1)
#' h <- 12
#' n <- length(AirPassengers)
#' z <- cbind(promo = rbinom(n + h, 1, 0.2),
#'            price = 50 + cumsum(rnorm(n + h)))
#' y <- AirPassengers + 30 * z[1:n, "promo"] - 2 * z[1:n, "price"]
#'
#' # Regressors with automatic lags: each starts from the same candidate lags
#' # as y (here 1 to 12, so not lag 0, the current value of the regressor) and
#' # lag selection keeps the useful ones
#' fit <- elm(y, xreg = z)
#' print(fit)
#' # Forecasting needs the future values of the regressors
#' frc <- forecast(fit, h = h, xreg = z)
#'
#' # Lags for each regressor, one vector per column of xreg, and lags kept
#' # whatever the lag selection says: promo at lag 0, price at lag 0
#' fit <- elm(y, xreg = z, xreg.lags = list(0, 0:2),
#'            xreg.keep = list(TRUE, c(TRUE, FALSE, FALSE)))
#'
#' # Use exactly the lags given, without lag selection
#' fit <- elm(y, xreg = z, xreg.lags = list(0, 1:3), sel.lag = FALSE)
#'
#' # Regressors as the only inputs: no lags of y (lags = 0)
#' fit <- elm(y, xreg = z, lags = 0, xreg.lags = list(0, 0:2), sel.lag = FALSE)
#'
#' # Number of hidden nodes (hd); by default it depends on the sample size
#' fit <- elm(AirPassengers, hd = 50)          # one hidden layer of 50 nodes
#' fit <- elm(AirPassengers, hd = c(30, 10))   # two random hidden layers
#' plot(fit)  # hidden nodes that the output layer does not use are dashed
#'
#' # Estimation of the output weights: "lasso" (default), "ridge", "step"
#' # (stepwise regression) or "lm"
#' fit <- elm(AirPassengers, type = "ridge")
#' fit <- elm(AirPassengers, type = "step")
#'
#' # Direct connections from the inputs to the output, besides the hidden layer
#' fit <- elm(AirPassengers, direct = TRUE)
#'
#' # The fast implementation, for many inputs; also returns an elm object
#' fit <- elm(AirPassengers, barebone = TRUE)
#'
#' # Save a fitted network and use it later
#' fit <- elm(AirPassengers)
#' file <- file.path(tempdir(), "elm_fit.rds")
#' saveRDS(fit, file)
#' fit <- readRDS(file)
#' frc <- forecast(fit, h = 12)
#'
#' # Apply a network to new data without re-estimating it: a network fitted up
#' # to 1958 is used on the full series
#' fit.old <- elm(window(AirPassengers, end = c(1958, 12)))
#' fit.new <- elm(AirPassengers, model = fit.old)
#' # ...or keep its specification and re-estimate on the new data (the number
#' # of hidden nodes is chosen again)
#' fit.new <- elm(AirPassengers, model = fit.old, retrain = TRUE)
#'
#' # A reused network with regressors needs the regressors of the new data
#' fit.old <- elm(window(y, end = c(1958, 12)), xreg = z, xreg.lags = list(0, 0:2))
#' fit.new <- elm(y, model = fit.old, xreg = z)
#' frc <- forecast(fit.new, h = h, xreg = z)
#' }
#'
#' @export elm

elm <- function(y, m = NULL, hd = NULL, type = c("lasso", "ridge", "step", "lm"), reps = 20,
                comb = c("median", "mean", "mode"), lags = NULL, keep = NULL, difforder = NULL,
                outplot = FALSE, sel.lag = TRUE, direct = FALSE, allow.det.season = TRUE,
                det.type = c("auto", "bin", "trg"), xreg = NULL, xreg.lags = NULL, xreg.keep = NULL,
                barebone = FALSE, model = NULL, retrain = FALSE) {

  type <- match.arg(type[1], c("lasso", "ridge", "step", "lm", "ls"))
  comb <- match.arg(comb)
  det.type <- match.arg(det.type)
  outplot <- as_flag(outplot, "outplot")
  sel.lag <- as_flag(sel.lag, "sel.lag")
  direct <- as_flag(direct, "direct")
  allow.det.season <- as_flag(allow.det.season, "allow.det.season")
  barebone <- as_flag(barebone, "barebone")
  retrain <- as_flag(retrain, "retrain")
  check_ts(y)
  ff <- resolve_periods(if (is.null(m) && !is.null(model)) model$ff else m, y)
  xreg <- check_xreg(xreg, xreg.lags, y)
  hd <- check_hd(hd)

  reuse <- !is.null(model)
  if (reuse) {
    check_reuse_model(model, "elm", xreg, ff)
    if (!retrain) hd <- model$hd
    comb <- model$comb
    direct <- model$direct
    type <- model$type
    reps <- length(model$b)
    barebone <- inherits(model, "elm.fast")
  }
  train <- !reuse || retrain

  pp <- prepare_data(y, ff, lags, keep, difforder, sel.lag, allow.det.season, det.type,
                     xreg, xreg.lags, xreg.keep, model)
  Y <- pp$Y
  X <- pp$X

  if (is.null(hd)) {
    hd <- min(100 - 60 * (type %in% c("step", "lm", "ls")),
              max(4, nrow(Y) - 2 - direct * ncol(X)))
  }
  if (barebone && train && length(hd) > 1) {
    stop("barebone = TRUE supports a single hidden layer only.", call. = FALSE)
  }

  # Train the members. The result holds everything elm_predict_member() needs.
  if (barebone) {
    if (train) {
      f.elm <- elm.fast(Y, X, hd = hd, reps = reps, comb = comb, type = type, direct = direct,
                        linscale = FALSE, output = "linear", core = TRUE)
      members <- c(f.elm[c("hd", "W.in", "W", "b", "W.dct")], list(net = NULL))
      Yhat.sc <- f.elm$fitted.all
    } else {
      members <- c(model[c("hd", "W.in", "W", "b", "W.dct")], list(net = NULL))
    }
  } else {
    net <- if (train) elm_random_layer(Y, X, hd, reps) else model$net
    members <- if (train) {
      elm_fit_output(net, Y, X, hd, reps, type, direct)
    } else {
      model[c("W", "b", "W.dct")]
    }
    members <- c(members, list(net = net, hd = hd, W.in = NULL))
  }
  members$direct <- direct
  members$fast <- barebone
  if (!(barebone && train)) {
    Yhat.sc <- predict_members(function(x, r) elm_predict_member(members, x, r), X, reps)
  }

  res <- finalise_fit(Yhat.sc, pp, y, comb)

  out <- structure(list("net" = members$net, "hd" = members$hd, "W.in" = members$W.in,
                        "W" = members$W, "b" = members$b, "W.dct" = members$W.dct,
                        "ff" = pp$ff, "lags" = pp$lags, "xreg.lags" = pp$xreg.lags, "difforder" = pp$difforder,
                        "sdummy" = pp$sdummy, "ff.det" = pp$ff.det, "det.type" = pp$det.type,
                        "y" = y, "minmax" = pp$minmax, "xreg.minmax" = pp$xreg.minmax,
                        "comb" = comb, "type" = type, "direct" = direct, "fitted" = res$fitted,
                        "MSE" = res$MSE, "fitted.all" = res$fitted.all),
                   class = if (barebone) c("elm", "elm.fast") else "elm")

  if (outplot) plot_fit(out)

  out
}

# ---------------------------------------------------------------------------
# ELM building blocks
# ---------------------------------------------------------------------------

# Random hidden layer(s). neuralnet is used only to draw random weights and to
# evaluate the logistic hidden units: the huge threshold stops training
# immediately, the output weights are then estimated by elm_output_weights().
elm_random_layer <- function(Y, X, hd, reps) {
  neuralnet::neuralnet(network_formula(X), cbind(Y, X), hidden = hd, threshold = 10^10,
                       rep = reps, err.fct = "sse", linear.output = FALSE)
}

# Activations of the last hidden layer of member r.
elm_hidden <- function(net, X, r, hd.last) {
  neurons <- neuralnet::compute(net, X, r)$neurons
  neurons[[length(neurons)]][, 1 + seq_len(hd.last), drop = FALSE]  # drop bias column
}

# Estimate output weights of every member of a neuralnet-based ELM.
# Members run in parallel if options(nnfor.cores) allows; their random
# cross-validation folds are drawn first, in member order, so results do not
# depend on the number of cores.
elm_fit_output <- function(net, Y, X, hd, reps, type, direct) {
  hd.last <- tail(hd, 1)
  folds <- lapply(seq_len(reps), function(r) elm_draw_folds(type, nrow(Y)))
  fits <- nnfor_lapply(seq_len(reps), function(r) {
    H <- elm_hidden(net, X, r, hd.last)
    elm_split_weights(elm.train(Y, if (direct) cbind(H, X) else H, type, foldid = folds[[r]]),
                      hd.last, direct, colnames(X))
  })
  W.dct <- vector("list", reps)
  if (direct) W.dct <- lapply(fits, `[[`, "W.dct")
  list(W = lapply(fits, `[[`, "W"), b = vapply(fits, `[[`, numeric(1), "b"), W.dct = W.dct)
}

# Cross-validation folds for the output layer, drawn as cv.glmnet() would
# draw them (10 folds). NULL when the estimation type does not use them.
elm_draw_folds <- function(type, n) {
  if (type %in% c("lasso", "ridge")) sample(rep(seq(10), length.out = n)) else NULL
}

# Split an output weight vector c(bias, hidden, direct) into its parts.
elm_split_weights <- function(w, hd, direct, x.names) {
  out <- list(b = w[1], W = w[1 + seq_len(hd), , drop = FALSE], W.dct = NULL)
  if (direct) {
    out$W.dct <- w[-(1:(1 + hd)), , drop = FALSE]
    if (!is.null(x.names)) rownames(out$W.dct) <- x.names
  }
  out
}

# Prediction of member r for the rows of x, on the scaled/differenced scale.
# Used for in-sample fit and for forecasting.
elm_predict_member <- function(members, x, r) {
  if (members$fast) {
    elm_fast_raw(x, members$W.in[[r]], members$W[[r]], members$b[r],
                 members$W.dct[[r]], members$direct)
  } else {
    H <- elm_hidden(members$net, x, r, tail(members$hd, 1))
    H %*% members$W[[r]] + members$b[r] + if (members$direct) x %*% members$W.dct[[r]] else 0
  }
}

# Estimate the output layer weights. Z holds the hidden activations (and the
# inputs when there are direct connections). Returns a one-column matrix
# c(bias, weights), with zero weights for unused regressors.
elm.train <- function(Y, Z, type, output = "linear", foldid = NULL) {
  family <- if (output == "logistic") "binomial" else "gaussian"
  if (type %in% c("lasso", "ridge")) {
    fit <- suppressWarnings(glmnet::cv.glmnet(Z, cbind(Y), alpha = as.numeric(type == "lasso"),
                                              family = family, foldid = foldid))
    cf <- as.vector(coef(fit))
    return(cbind(cf))
  }
  # Least squares, optionally with stepwise selection
  k <- ncol(Z)
  reg.data <- as.data.frame(cbind(Y, Z))
  colnames(reg.data) <- c("Y", paste0("X", 1:k))
  # Take care of linear dependency
  aliased <- rownames(alias(as.formula(paste0("Y~", paste0("X", 1:k, collapse = "+"))),
                            data = reg.data)$Complete)
  frm <- as.formula(paste0("Y~", paste0(setdiff(colnames(reg.data)[-1], aliased), collapse = "+")))
  fit <- suppressWarnings(lm(frm, reg.data))
  if (type == "step") {
    fit <- suppressWarnings(MASS::stepAIC(fit, trace = 0))
  }
  cf.fit <- coef(fit)
  cf <- rep(0, k + 1)
  cf[1] <- cf.fit[1]
  cf[match(names(cf.fit)[-1], colnames(reg.data))] <- cf.fit[-1]
  cbind(cf)
}
