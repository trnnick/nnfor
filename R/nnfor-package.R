#' nnfor: Time Series Forecasting with Neural Networks
#'
#' The \pkg{nnfor} package provides automatic time series modelling with neural networks. It facilitates fully automatic, semi-manual or fully manual specification of networks, using multilayer perceptrons (\code{\link{mlp}}) and extreme learning machines (\code{\link{elm}}). Specification covers the input lags, differencing, deterministic seasonality, exogenous regressors and the number of hidden nodes, and forecasts are combined across an ensemble of networks.
#'
#' @section Training backends:
#' Multilayer perceptrons are trained with \code{\link[neuralnet]{neuralnet}}
#' by default, which is installed with \pkg{nnfor}. As an option, they can be
#' trained with \href{https://torch.mlverse.org}{torch} by setting
#' \code{backend = "torch"} in \code{\link{mlp}}. torch adds modern activation
#' functions and optimizers, custom architectures (\code{hd} given as a
#' function), and trains all networks of an ensemble simultaneously. It is
#' optional: install it with \code{install.packages("torch")} followed by
#' \code{torch::install_torch()}. Nothing else in \pkg{nnfor} needs it.
#'
#' By default the torch backend reproduces neuralnet's training setup and
#' accepts neuralnet's argument names; the two backends still produce
#' different networks. The torch settings, suggested settings and the
#' differences between the backends are described in the Details of
#' \code{\link{mlp}}, and \code{\link{compare_backends}} compares them on a
#' series. Extreme learning machines (\code{\link{elm}},
#' \code{\link{elm.fast}}) estimate output weights by regression and do not
#' use a training backend.
#'
#' @section Functions:
#' \subsection{Modelling}{
#' \describe{
#' \item{\code{\link{mlp}}}{Multilayer perceptrons, trained with neuralnet or torch.}
#' \item{\code{\link{elm}}}{Extreme learning machines.}
#' \item{\code{\link{elm.fast}}}{Fast extreme learning machine for general regression
#'   problems (inputs and target given directly); the engine of
#'   \code{elm(barebone = TRUE)}.}
#' }
#' Fitted networks can be printed, and reused on new data with the \code{model}
#' argument of \code{mlp} and \code{elm}.
#' }
#'
#' \subsection{Forecasting}{
#' \describe{
#' \item{\code{\link[=forecast.mlp]{forecast}(fit)}}{Forecasts from \code{mlp} and \code{elm} fits
#'   (\code{\link{forecast.mlp}}, \code{\link{forecast.elm}}).}
#' \item{\code{\link[=predict.elm.fast]{predict}(fit, newx)}}{Predictions from \code{elm.fast} fits for new inputs.}
#' \item{\code{\link{mlp.thief}}, \code{\link{elm.thief}}}{Wrappers to forecast with temporal
#'   hierarchies using the \pkg{thief} package.}
#' }
#' }
#'
#' \subsection{Visualisation}{
#' \describe{
#' \item{\code{\link[=plot.mlp]{plot}(fit)}}{For \code{mlp} and \code{elm} fits: the network
#'   architecture or the in-sample fit (\code{\link{plot.mlp}}, \code{\link{plot.elm}}).}
#' \item{\code{\link{plot_network}}, \code{\link{plot_fit}}}{The same plots, with all their
#'   options for customisation.}
#' \item{\code{\link[=plot.forecast.net]{plot}(forecast(fit))}}{Forecasts from \code{mlp} and \code{elm} fits.}
#' \item{\code{\link[=plot.elm.fast]{plot}(fit)}}{For \code{elm.fast} fits: fitted against actual values.}
#' }
#' All plots respect graphical parameters set by the user and accept further
#' graphical arguments.
#' }
#'
#' \subsection{Transparency and comparison}{
#' \describe{
#' \item{\code{\link{nnfor_map}}}{Map of how information flows through the package when
#'   fitting or forecasting, optionally annotated with a fitted network; see
#'   \code{\link{plot.nnfor_map}}.}
#' \item{\code{\link{compare_backends}}}{Fits and forecasts a series with several backend
#'   configurations, reporting run times, in-sample, and out-of-sample errors.}
#' }
#' }
#'
#' \subsection{Utilities}{
#' \describe{
#' \item{\code{\link{linscale}}}{Linear scaling of a vector to a given range, and its reverse.}
#' }
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
#' @note You can find a tutorial how to use the package \href{https://kourentzes.com/forecasting/2019/01/16/tutorial-for-the-nnfor-r-package/}{here}.
#'
#' @references
#' \itemize{
#' \item For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{https://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.
#' \item For ensemble combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.
#' \item For variable selection see: Crone S.F., Kourentzes N. (2010) \href{https://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction -- A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.
#' }
#'
#' @keywords package ts
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#'
#' @importFrom graphics abline arrows axis legend lines plot points par rect text strwidth strheight
#' @importFrom stats alias as.formula coef deltat density end frequency friedman.test lm median runif start time ts predict setNames
#' @importFrom utils head tail
#' @importFrom forecast nsdiffs
#' @importFrom generics forecast
#' @importFrom tsutils cmav lagmatrix seasdummy mseastest
#' @importFrom uroot ch.test
"_PACKAGE"
