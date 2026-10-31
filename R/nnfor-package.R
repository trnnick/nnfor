#' nnfor: Time Series Forecasting with Neural Networks
#'
#' The \pkg{nnfor} package provides automatic time series modelling with neural networks. It facilitates fully automatic, semi-manual or fully manual specification of networks, using multilayer perceptrons (\code{\link{mlp}}) and extreme learning machines (\code{\link{elm}}).
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
#' \item{For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{https://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.}
#' \item{For ensemble combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.}
#' \item{For variable selection see: Crone S.F., Kourentzes N. (2010) \href{https://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction – A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.}
#' }
#'
#' @keywords package ts
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#'
#' @importFrom graphics arrows axis lines plot points par rect text strwidth strheight
#' @importFrom stats alias as.formula coef deltat density end frequency friedman.test lm median runif start time ts predict setNames
#' @importFrom utils head tail
#' @importFrom forecast nsdiffs
#' @importFrom generics forecast
#' @importFrom tsutils cmav lagmatrix seasdummy mseastest
#' @importFrom uroot ch.test
"_PACKAGE"
