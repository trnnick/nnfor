
#' @rdname forecast.elm
#' @title Forecast using ELM neural network.
#' @description Create forecasts using ELM neural networks.
#'
#' @param object ELM network object, produced using \code{\link{elm}}.
#' @param h Forecast horizon. If NULL then h is set to match frequency of time series.
#' @param y Optionally forecast using different data than what the network was trained on. Expected to create havoc and do really bad things!
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set plus the forecast horizon, but can be longer. Set it to NULL if no xreg inputs are used.
#' @param ... Unused argument.
#'
#' @return An object of classes "\code{forecast.net}" and "\code{forecast}".
#'   The function \code{plot} produces a plot of the forecasts.
#'   An object of class \code{"forecast.net"} is a list containing the following elements:
#' \itemize{
#' \item{\code{method}{ - The name of the forecasting method as a character string}}
#' \item{\code{mean}{ - Point forecasts as a time series}}
#' \item{\code{all.mean}{ - An array h x reps of all ensemble members forecasts, where reps are the number of ensemble members.}}
#' \item{\code{x}{ - The original time series used to create the network.}}
#' \item{\code{fitted}{ - Fitted values.}}
#' \item{\code{residuals}{ - Residuals from the fitted network.}}
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}, \code{\link{elm.thief}}, \code{\link{mlp}}.
#' @keywords elm thief ts
#'
#' @examples
#' \dontshow{
#'  fit <- elm(AirPassengers,reps=1)
#'  frc <- forecast(fit,h=36)
#'  print(frc)
#'  plot(frc)
#' }
#' \dontrun{
#'  fit <- elm(AirPassengers)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#'
#' @export
forecast.elm <- function(object, h = NULL, y = NULL, xreg = NULL, ...) {
  forecast_network(object, h = h, y = y, xreg = xreg)
}

#' @export
predict.elm <- function(object, ...) {
  warning("Use forecast() instead of predict() for ELM.")
  forecast.elm(object, ...)
}

#' @rdname forecast.mlp
#' @title Forecast using MLP neural network.
#' @description  Create forecasts using MLP neural networks.
#'
#' @param object MLP network object, produced using \code{\link{mlp}}.
#' @param h Forecast horizon. If NULL then h is set to match frequency of time series.
#' @param y Optionally forecast using different data than what the network was trained on. Expected to create havoc and do really bad things!
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set plus the forecast horizon, but can be longer. Set it to NULL if no xreg inputs are used.
#' @param ... Unused argument.
#'
#' @inherit forecast.elm return
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{mlp}}, \code{\link{mlp.thief}}, \code{\link{elm}}.
#' @keywords mlp thief ts
#'
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers,reps=1)
#'  frc <- forecast(fit,h=36)
#'  print(frc)
#'  plot(frc)
#' }
#' \dontrun{
#'  fit <- mlp(AirPassengers)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#'
#' @export
forecast.mlp <- function(object, h = NULL, y = NULL, xreg = NULL, ...) {
  forecast_network(object, h = h, y = y, xreg = xreg)
}

#' @export
predict.mlp <- function(object, ...) {
  warning("Use forecast() instead of predict() for MLP.")
  forecast.mlp(object, ...)
}

# ---------------------------------------------------------------------------
# Shared forecasting engine
# ---------------------------------------------------------------------------

# Number of ensemble members of a fitted network.
n_members <- function(object) {
  if (inherits(object, "elm")) length(object$b) else object_backend(object)$n_members(object$net)
}

# A function(x, r) giving the scaled prediction of member r for input rows x.
member_predictor <- function(object) {
  if (inherits(object, "elm")) {
    members <- object[c("net", "hd", "W.in", "W", "b", "W.dct", "direct")]
    members$fast <- inherits(object, "elm.fast")
    function(x, r) elm_predict_member(members, x, r)
  } else {
    backend <- object_backend(object)
    function(x, r) backend$predict(object$net, x, r)
  }
}

# Recursive multi-step forecasts from an mlp or elm object.
forecast_network <- function(object, h = NULL, y = NULL, xreg = NULL) {
  y <- y %||% object$y
  h <- h %||% max(object$ff %||% get_frequencies(y))
  if (length(h) != 1 || h < 1 || h %% 1 != 0) {
    stop("Forecast horizon h must be a positive integer.", call. = FALSE)
  }
  fstart <- forecast_start(y)
  difforder <- object$difforder
  lags <- object$lags
  xreg.lags <- object$xreg.lags

  # Regressors
  if (is.null(xreg)) {
    if (sum(lengths(xreg.lags)) > 0) {
      stop("The network uses xreg inputs; provide xreg covering the sample and forecast horizon.",
           call. = FALSE)
    }
    xreg.lags <- NULL
  } else {
    xreg <- as.matrix(xreg)
    if (length(xreg.lags) != ncol(xreg)) {
      stop("Number of xreg inputs is not consistent with network specification (number of xreg.lags).",
           call. = FALSE)
    }
    if (nrow(xreg) < length(y) + h) {
      stop("Length of xreg must be longer that y + forecast horizon.", call. = FALSE)
    }
    xreg <- rescale_xreg(xreg, object$xreg.minmax, sum(difforder))
  }

  # Target on the network's scale
  y.d <- difference_series(y, difforder)
  Y <- as.vector(linscale(y.d[[length(y.d)]], minmax = object$minmax)$x)
  n <- length(Y)

  Xd <- if (isTRUE(object$sdummy)) {
    season_dummies_forecast(h, object$ff.det, object$det.type, y)
  } else NULL

  predict_fun <- member_predictor(object)
  reps <- n_members(object)
  Yfrc <- array(NA, c(h, reps), dimnames = list(paste0("t+", 1:h), paste0("NN.", 1:reps)))

  for (r in 1:reps) {
    path <- c(Y, numeric(h))  # history extended with this member's forecasts
    for (i in 1:h) {
      inp <- lag_inputs(path, xreg, lags, xreg.lags, n + i)
      xi <- cbind(inp$X, cbind_list(inp$Xreg), Xd[i, , drop = FALSE])
      path[n + i] <- predict_fun(xi, r)
    }
    frc <- linscale(path[n + 1:h], object$minmax, rev = TRUE)$x
    Yfrc[, r] <- undifference_forecast(frc, y.d, difforder)
  }

  fout <- ts(frc.comb(Yfrc, object$comb), frequency = frequency(y), start = fstart)
  out <- list("method" = class(object), "mean" = fout,
              "all.mean" = ts(Yfrc, frequency = frequency(y), start = fstart),
              "x" = y, "fitted" = object$fitted, "residuals" = y - object$fitted)
  structure(out, class = c("forecast.net", "forecast"))
}
