#' @title MLP network for THieF.
#' @description Function for MLP forecasting with Temporal Hierarchies.
#'
#' @param y Input time series. Can be ts or msts object.
#' @param h Forecast horizon. If NULL then h is set to match frequency of time series.
#' @param ... Additional arguments passed to \code{\link{mlp}}.
#'
#' @return An object of classes "\code{forecast.net}" and "\code{forecast}".
#'   The function \code{plot} produces a plot of the forecasts.
#'   An object of class \code{"forecast.net"} is a list containing the following elements:
#' \itemize{
#' \item \code{method}: The name of the forecasting method as a character string
#' \item \code{mean}: Point forecasts as a time series
#' \item \code{all.mean}: An array h x reps of all ensemble members forecasts, where reps are the number of ensemble members.
#' \item \code{x}: The original time series (either \code{fit} used to create the network.
#' \item \code{fitted}: Fitted values. Any values not fitted for the initial period of the time series are imputted with NA.
#' \item \code{residuals}: Residuals from the fitted network.
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{mlp}}, \code{\link{elm.thief}}.
#' @references
#' \itemize{
#'   \item For forecasting with temporal hierarchies see: Athanasopoulos G., Hyndman R.J., Kourentzes N., Petropoulos F. (2017) \href{https://kourentzes.com/forecasting/2017/02/27/forecasting-with-temporal-hierarchies-3/}{Forecasting with Temporal Hierarchies}. \emph{European Journal of Operational research}, \bold{262}(\bold{1}), 60-74.
#'   \item For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.
#' }
#' @note This function is created to work with Temporal Hierarchies (\href{https://cran.r-project.org/package=thief}{thief} package). For conventional MLP networks use \code{\link{mlp}}.
#' @examples
#' \dontrun{
#'   library(thief)
#'   frc <- thief(AirPassengers,forecastfunction=mlp.thief)
#'   plot(frc)
#' }
#'
#' @keywords mlp thief ts
#' @export mlp.thief

mlp.thief <- function(y, h = NULL, ...) {
  net_thief(mlp, y, h, ...)
}

#' @title ELM network for THieF.
#' @description Function for ELM forecasting with Temporal Hierarchies.
#'
#' @param y Input time series. Can be ts or msts object.
#' @param h Forecast horizon. If NULL then h is set to match frequency of time series.
#' @param ... Additional arguments passed to \code{\link{elm}}.
#'
#' @inherit mlp.thief return references
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}, \code{\link{mlp.thief}}.
#' @note This function is created to work with Temporal Hierarchies (\href{https://cran.r-project.org/package=thief}{thief} package). For conventional ELM networks use \code{\link{elm}}.
#' @examples
#' \dontrun{
#'   library(thief)
#'   frc <- thief(AirPassengers,forecastfunction=elm.thief)
#'   plot(frc)
#' }
#'
#' @keywords elm thief ts
#' @export elm.thief
#'
elm.thief <- function(y, h = NULL, ...) {
  net_thief(elm, y, h, ...)
}

# Fit a network and forecast in the form expected by thief::thief().
net_thief <- function(fit_fun, y, h = NULL, ...) {
  args <- list(...)
  args$level <- NULL  # passed by thief, not used by the networks
  args$y <- y
  fit <- do.call(fit_fun, args)

  out <- forecast(fit, h = h %||% frequency(y), xreg = args[["xreg"]])
  # Make fitted values span the complete sample
  n <- length(out$x)
  m <- length(out$fitted)
  if (m < n) {
    out$fitted <- ts(c(rep(NA, n - m), out$fitted), frequency = frequency(out$fitted),
                     end = end(out$fitted))
  }
  out
}
