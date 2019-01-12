#' @references
#' \itemize{
#' \item{For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{http://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.}
#' \item{For ensemble combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{http://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.}
#' \item{For variable selection see: Crone S.F., Kourentzes N. (2010) \href{http://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction – A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.}
#' }
#'
#' @importFrom grDevices rainbow
#' @importFrom graphics arrows axis lines plot points text hist
#' @importFrom stats alias as.formula coef cor.test deltat end frequency friedman.test lm median runif start time ts ts.plot density fft uniroot
#' @importFrom utils head tail
#' @importFrom forecast forecast
#' @importFrom tsutils cmav lagmatrix seasdummy mseastest
NULL
