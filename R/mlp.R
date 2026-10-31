#' Multilayer Perceptron for time series forecasting
#'
#' This function fits MLP neural networks for time series forecasting.
#'
#' @param y Input time series. Can be ts or msts object.
#' @param m Seasonal period(s) of the series. Use a vector for multiple seasonalities, e.g. \code{c(7, 365.25)}. If NULL (default) they are taken from \code{y}: its frequency for a ts object, or all seasonal periods for an msts object. When given, \code{m} overrides the periods of \code{y} for lag specification, seasonality and differencing tests and seasonal dummies.
#' @param hd Number of hidden nodes. This can be a vector, where each number represents the number of hidden nodes of a different hidden layer. With \code{backend = "torch"} it can also be a function defining a custom architecture, see Details.
#' @param reps Number of networks to train, the result is the ensemble forecast.
#' @param comb Combination operator for forecasts when reps > 1. Can be "median", "mode" (based on KDE estimation) and "mean".
#' @param lags Lags of y to use as inputs. If none provided then 1:frequency(y) is used. Use 0 for no univariate lags.
#' @param keep Logical vector to force lags to stay in the model if sel.lag == TRUE. If NULL then it keep = rep(FALSE,length(lags)).
#' @param difforder Vector including the differencing lags. For example c(1,12) will apply first and seasonal (12) differences. For no differencing use 0. For automatic selection use NULL.
#' @param outplot Provide plot of model fit. Can be TRUE or FALSE. For a customised plot use \code{plot(fit, which = "fit", ...)} after fitting.
#' @param sel.lag Automatically select lags. Can be TRUE or FALSE.
#' @param allow.det.season Permit modelling seasonality with deterministic dummies.
#' @param det.type Type of deterministic seasonality dummies to use. This can be "bin" for binary or "trg" for a sine-cosine pair. With "auto" if ony a single seasonality is used and periodicity is up to 12 then "bin" is used, otherwise "trg".
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set, but can be longer.
#' @param xreg.lags This is a list containing the lags for each exogenous variable. Each list is a numeric vector containing lags. If xreg has 3 columns then the xreg.lags list must contain three elements. If NULL then it is automatically specified.
#' @param xreg.keep List of logical vectors to force lags of xreg to stay in the model if sel.lag == TRUE. If NULL then all exogenous lags can be removed. The syntax for multiple xreg is the same as for xreg.lags.
#' @param hd.auto.type Used only if hd==NULL. "set" fixes hd=5. "valid" uses a 20\% validation set (randomly) sampled to find the best number of hidden nodes. "cv" uses 5-fold cross-validation. "elm" uses ELM to estimate the number of hidden nodes (experimental).
#' @param hd.max When hd.auto.type is set to either "valid" or "cv" then this argument can be used to set the maximum number of hidden nodes to evaluate, otherwise the maximum is set automatically.
#' @param backend Library used to train the networks: "neuralnet" (default) or "torch", which needs the \href{https://cran.r-project.org/package=torch}{torch} package. See Details.
#' @param model A previously trained mlp object. If this is provided then the same model is fitted to y, without re-estimating any model parameters.
#' @param retrain If a previous model is provided, retrain the network or not.
#' @param ... Settings for training, passed to the backend: arguments of \code{\link[neuralnet]{neuralnet}} for \code{backend = "neuralnet"}, or the torch settings listed in Details. They are also used when searching for the number of hidden nodes.
#'
#' @details
#' \subsection{Training backends}{
#' Networks are trained with \code{\link[neuralnet]{neuralnet}} by default.
#' With \code{backend = "torch"} they are trained with
#' \href{https://torch.mlverse.org}{torch}, which supports modern activation
#' functions and optimizers and custom architectures. torch is optional: install
#' it with \code{install.packages("torch")} followed by \code{torch::install_torch()}.
#' Use \code{\link{compare_backends}} to compare them on a series, and
#' \code{\link{nnfor_map}} to see where training sits in the modelling steps.
#' }
#'
#' \subsection{torch settings}{
#' The torch backend accepts neuralnet's argument names, and by default trains
#' as neuralnet does: logistic hidden layers, linear output, full-batch Rprop on
#' the sum of squared errors, stopping when the largest absolute gradient falls
#' below \code{threshold} or after \code{stepmax} steps. Settings, passed in \code{...}:
#' \describe{
#' \item{\code{act.fct}}{Hidden layer activation: \code{"logistic"} (default, also
#'   \code{"sigmoid"}), \code{"tanh"}, \code{"relu"}, \code{"leaky_relu"},
#'   \code{"elu"}, \code{"selu"}, \code{"gelu"}, \code{"silu"} (also
#'   \code{"swish"}), \code{"mish"}, \code{"softplus"}, or a function of a torch tensor.}
#' \item{\code{algorithm}}{Optimizer: \code{"rprop"} (default; neuralnet's
#'   \code{"rprop+"} and \code{"rprop-"} are accepted), \code{"adam"},
#'   \code{"adamw"}, \code{"sgd"} (also \code{"backprop"}), \code{"rmsprop"} or
#'   \code{"adagrad"}. L-BFGS is not offered: it couples the ensemble members,
#'   which are trained together.}
#' \item{\code{learningrate}}{Learning rate. Defaults: 0.1 as the initial step
#'   of Rprop (as neuralnet), 0.001 for Adam and AdamW, 0.01 otherwise.}
#' \item{\code{threshold}, \code{stepmax}}{Stopping rule, as in neuralnet
#'   (defaults 0.01 and 1e5).}
#' \item{\code{learningrate.limit}, \code{learningrate.factor}}{Rprop step
#'   limits and factors, as in neuralnet (defaults \code{list(min = 1e-10, max = 0.1)}
#'   and \code{list(minus = 0.5, plus = 1.2)}).}
#' }
#' Popular choices for shallow networks: \code{"tanh"} is a smooth, zero-centred
#' replacement for the logistic function; \code{"silu"} and \code{"gelu"} are
#' smooth alternatives to \code{"relu"}, which with few hidden nodes can leave
#' nodes inactive; Adam with a learning rate between 0.001 and 0.01 is the usual
#' default optimizer, and Rprop remains a good choice for full-batch training of
#' small networks.
#' }
#'
#' \subsection{Suggested torch settings}{
#' The defaults reproduce neuralnet's setup, and the function defaults are kept
#' for both backends. For torch we suggest
#' \code{act.fct = "gelu", algorithm = "adam", learningrate = 0.01, stepmax = 3000}.
#' On eight standard series (AirPassengers, co2, nottem, USAccDeaths, UKgas,
#' ldeaths, Nile, lynx; last season or 10 years held out, 10 networks per
#' ensemble), this was the most consistent of the settings tried: its
#' out-of-sample MSE was on average 6\% below neuralnet's, lower on five
#' series and at most 45\% higher on the others, for a similar run time.
#' \code{"relu"} and \code{"silu"} were less reliable, failing badly on some
#' series, and Rprop with \code{"tanh"} was slow to converge. Results vary
#' from series to series: \code{\link{compare_backends}} runs this kind of
#' comparison on your data. The default torch setup (neuralnet's settings)
#' was the fastest overall.
#' }
#'
#' \subsection{Custom architectures}{
#' With \code{backend = "torch"}, \code{hd} can be a function that receives the
#' number of inputs and returns the hidden layers as a torch module. nnfor
#' builds the inputs, adds a linear output layer, trains the ensemble and
#' produces forecasts. The number of hidden nodes is then not searched, so
#' \code{hd.auto.type} is ignored, and \code{act.fct} is not used: activations
#' are part of the module. For example, two hidden layers with dropout:
#' \preformatted{
#' arch <- function(n_inputs) {
#'   torch::nn_sequential(
#'     torch::nn_linear(n_inputs, 16), torch::nn_silu(), torch::nn_dropout(0.1),
#'     torch::nn_linear(16, 8), torch::nn_silu())
#' }
#' fit <- mlp(AirPassengers, hd = arch, backend = "torch", algorithm = "adam",
#'            learningrate = 0.01, stepmax = 3000)
#' }
#' or a module of your own, here with a skip connection:
#' \preformatted{
#' skip_block <- torch::nn_module(
#'   initialize = function(n_inputs, width = 8) {
#'     self$hidden <- torch::nn_linear(n_inputs, width)
#'     self$skip <- torch::nn_linear(n_inputs, width)
#'   },
#'   forward = function(x) torch::torch_tanh(self$hidden(x)) + self$skip(x))
#' fit <- mlp(AirPassengers, hd = function(n) skip_block(n), backend = "torch",
#'            algorithm = "adam", learningrate = 0.01, stepmax = 2000)
#' }
#' Ensemble members of custom architectures share one optimizer and one
#' backward pass, but run separate forward passes, so they are slower to
#' train than standard networks.
#' }
#'
#' \subsection{Differences between the backends}{
#' \itemize{
#' \item The two backends do not produce the same networks: torch computes in
#'   single precision (float32), and its Rprop does not use neuralnet's weight
#'   backtracking.
#' \item neuralnet discards ensemble members that do not converge within
#'   \code{stepmax}. torch keeps them, since with optimizers such as Adam the
#'   gradient may never fall below \code{threshold}; the number that converged
#'   is shown by \code{print}.
#' \item Ensemble members of standard networks are trained simultaneously as
#'   one batch of tensors, which gives the same result as training them one
#'   by one. Training time therefore grows little with \code{reps}, while
#'   neuralnet's grows in proportion; \code{options(nnfor.cores)} is not used
#'   by torch.
#' \item Results are reproducible with \code{set.seed()}: starting weights are
#'   drawn from R's random number generator, as neuralnet draws them, and
#'   torch's own generator is seeded from it.
#' \item Networks trained with torch need torch to forecast, also when loaded
#'   from a saved file.
#' \item Training runs on the CPU. GPUs offer little for networks of this size;
#'   support may be added in a later version.
#' }
#' }
#'
#' @return Return object of class \code{mlp}.
#'   The function \code{plot} produces a plot the network architecture or of the fit (see \code{\link{plot.mlp}}).
#'   \code{mlp} contains:
#' \itemize{
#' \item \code{net}: MLP networks.
#' \item \code{hd}: Number of hidden nodes.
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
#' \item \code{fitted}: Fitted values.
#' \item \code{MSE}: In-sample Mean Squared Error.
#' \item \code{MSEH}: If \code{hd.auto.type} is set to either "valid" or "cv" an array of the MSE error for each network size is provided. Otherwise this is NULL.
#' \item \code{fitted.all}: Fitted values of each ensemble member.
#' \item \code{backend}: Name of the library used to train the networks: "neuralnet" or "torch".
#' }
#' @section Parallel training:
#' Ensemble members can be trained in parallel with
#' \code{options(nnfor.cores = 4)} (forked workers on Linux and macOS) or
#' \code{options(nnfor.cores = cl)} with a cluster from
#' \code{parallel::makeCluster()}, which is recommended on Windows. All random
#' numbers are drawn before members are dispatched, so results are identical
#' for any number of cores and reproducible with \code{set.seed()}.
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{forecast.mlp}}, \code{\link{mlp.thief}}, \code{\link{elm}}.
#' @references
#' \itemize{
#' \item For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{https://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.
#' \item For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.
#' \item For variable selection see: Crone S.F., Kourentzes N. (2010) \href{https://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction -- A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.
#' }
#' @keywords mlp thief ts
#' @note To use mlp with Temporal Hierarchies (\href{https://cran.r-project.org/package=thief}{thief} package) see \code{\link{mlp.thief}}.
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers,reps=1)
#'  print(fit)
#'  plot(fit)
#' }
#' \dontrun{
#'  fit <- mlp(AirPassengers)
#'  print(fit)
#'  plot(fit)
#'  plot(fit, which = "fit")
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()) {
#'   # torch, with neuralnet's settings
#'   fit <- mlp(AirPassengers, reps = 5, backend = "torch")
#'   print(fit)
#'   # torch, with the suggested settings
#'   fit <- mlp(AirPassengers, reps = 5, backend = "torch", act.fct = "gelu",
#'              algorithm = "adam", learningrate = 0.01, stepmax = 3000)
#'   plot(forecast(fit, h = 24))
#'   # A custom architecture
#'   arch <- function(n_inputs) {
#'     torch::nn_sequential(torch::nn_linear(n_inputs, 16), torch::nn_silu(),
#'                          torch::nn_linear(16, 8), torch::nn_silu())
#'   }
#'   fit <- mlp(AirPassengers, reps = 5, hd = arch, backend = "torch",
#'              algorithm = "adam", learningrate = 0.01, stepmax = 1000)
#'   plot(fit)
#' }
#' }
#'
#' @export mlp

mlp <- function(y, m = NULL, hd = NULL, reps = 20, comb = c("median", "mean", "mode"),
                lags = NULL, keep = NULL, difforder = NULL, outplot = FALSE, sel.lag = TRUE,
                allow.det.season = TRUE, det.type = c("auto", "bin", "trg"),
                xreg = NULL, xreg.lags = NULL, xreg.keep = NULL,
                hd.auto.type = c("set", "valid", "cv", "elm"),
                hd.max = NULL, backend = c("neuralnet", "torch"), model = NULL, retrain = FALSE,
                ...) {

  comb <- match.arg(comb)
  det.type <- match.arg(det.type)
  hd.auto.type <- match.arg(hd.auto.type)
  backend <- match.arg(backend)
  outplot <- as_flag(outplot, "outplot")
  sel.lag <- as_flag(sel.lag, "sel.lag")
  allow.det.season <- as_flag(allow.det.season, "allow.det.season")
  retrain <- as_flag(retrain, "retrain")
  check_ts(y)
  ff <- resolve_periods(if (is.null(m) && !is.null(model)) model$ff else m, y)
  xreg <- check_xreg(xreg, xreg.lags, y)
  hd <- check_hd(hd, if (is.null(model)) backend else model$backend %||% "neuralnet")
  if (!is.null(hd) && hd.auto.type != "set") {
    warning("hd.auto.type is ignored because hd is given.", call. = FALSE)
  }

  reuse <- !is.null(model)
  if (reuse) {
    check_reuse_model(model, "mlp", xreg, ff)
    backend <- object_backend(model)
    hd <- model$hd
    comb <- model$comb
    reps <- backend$n_members(model$net)
  } else {
    backend <- nn_backend(backend)
  }

  pp <- prepare_data(y, ff, lags, keep, difforder, sel.lag, allow.det.season, det.type,
                     xreg, xreg.lags, xreg.keep, model)

  # Number of hidden nodes
  mseH <- NULL
  if (is.null(hd)) {
    hd <- switch(hd.auto.type,
                 "set" = 5,
                 "valid" = , "cv" = {
                   res <- select_hidden_cv(pp$Y, pp$X, comb, reps, hd.auto.type, hd.max, backend, ...)
                   mseH <- res$mseH
                   res$hd
                 },
                 "elm" = select_hidden_elm(pp$Y, pp$X))
  }

  # Train (or reuse) the networks
  # When retraining a reused model without new settings, keep its settings.
  train.args <- list(...)
  if (reuse && !length(train.args)) train.args <- model$net$settings
  net <- if (!reuse || retrain) {
    do.call(backend$train, c(list(pp$Y, pp$X, hd, reps), train.args))
  } else {
    model$net
  }
  reps <- backend$n_members(net)  # some members may have failed to train

  Yhat.sc <- predict_members(function(X, r) backend$predict(net, X, r), pp$X, reps)
  res <- finalise_fit(Yhat.sc, pp, y, comb)

  out <- structure(list("net" = net, "hd" = hd, "ff" = pp$ff, "lags" = pp$lags, "xreg.lags" = pp$xreg.lags,
                        "difforder" = pp$difforder, "sdummy" = pp$sdummy, "ff.det" = pp$ff.det,
                        "det.type" = pp$det.type, "y" = y, "minmax" = pp$minmax,
                        "xreg.minmax" = pp$xreg.minmax, "comb" = comb, "fitted" = res$fitted,
                        "MSE" = res$MSE, "MSEH" = mseH, "fitted.all" = res$fitted.all,
                        "backend" = backend$name),
                   class = "mlp")

  if (outplot) plot_fit(out)

  out
}

# ---------------------------------------------------------------------------
# Automatic specification of the number of hidden nodes
# ---------------------------------------------------------------------------

# Validation ("valid") or K-fold cross-validation ("cv") over 1:hd.max nodes
# for a single hidden layer, scored by one-step-ahead MSE of the ensemble.
select_hidden_cv <- function(Y, X, comb, reps, type = c("cv", "valid"), hd.max = NULL,
                             backend = nn_backend("neuralnet"), ...) {
  type <- match.arg(type)
  K <- 5                                   # Number of folds
  val.size <- 0.2                          # Size of validation set
  reps <- min(20, max(2, reps))            # Ensemble size, at most 20
  n <- nrow(Y)
  if (is.null(hd.max)) {
    hd.max <- max(2, min(ncol(X) + 2, n - 2))
  }

  if (type == "cv") {
    if (K >= n) {
      stop("Too few observations to perform cross-validation for specification of hidden nodes.",
           call. = FALSE)
    }
    perm <- sample(1:n)
    cuts <- seq(0, n, length.out = K + 1)
    folds <- lapply(1:K, function(i) which(perm <= cuts[i + 1] & perm > cuts[i]))
  } else {
    folds <- list(sample(1:n, round(val.size * n)))
  }

  err.h <- array(NA, c(hd.max, 1), dimnames = list(paste0("H.", 1:hd.max), "MSE"))
  for (h in 1:hd.max) {
    err.cv <- vapply(folds, function(tst) {
      trn <- setdiff(1:n, tst)
      net <- backend$train(Y[trn, , drop = FALSE], X[trn, , drop = FALSE], h, reps, ...)
      X.tst <- X[tst, , drop = FALSE]
      frc <- predict_members(function(x, r) backend$predict(net, x, r), X.tst,
                             backend$n_members(net))
      mean((Y[tst, , drop = FALSE] - frc.comb(frc, comb))^2)
    }, numeric(1))
    err.h[h] <- mean(err.cv)
  }

  list("hd" = which.min(err.h), "mseH" = err.h)
}

# Estimate the number of useful hidden nodes from an oversized ELM: count the
# hidden nodes that remain significant after stepwise regression, take the
# median over 20 random hidden layers.
select_hidden_elm <- function(Y, X) {
  sz <- min(ncol(X) + 2, nrow(Y) - 2)
  reps <- 20
  net <- elm_random_layer(Y, X, sz, reps)
  hd <- unlist(nnfor_lapply(1:reps, function(r) {
    Z <- elm_hidden(net, X, r, sz)
    reg.data <- as.data.frame(cbind(Y, Z))
    colnames(reg.data) <- c("Y", paste0("X", 1:sz))
    # Take care of linear dependency
    aliased <- rownames(alias(Y ~ ., data = reg.data)$Complete)
    frm <- as.formula(paste0("Y~", paste0(setdiff(colnames(reg.data)[-1], aliased), collapse = "+")))
    fit <- suppressWarnings(lm(frm, reg.data))
    fit <- suppressWarnings(MASS::stepAIC(fit, trace = 0, direction = "backward"))
    p <- summary(fit)$coefficients[, 4]
    sum(p < 0.05, na.rm = TRUE) - (p[1] < 0.05)
  }))
  max(1, round(median(hd)))
}
