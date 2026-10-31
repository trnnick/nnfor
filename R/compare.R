#' Compare network backends
#'
#' Fits \code{\link{mlp}} with each backend configuration on a time series,
#' holding out the last \code{h} observations, and reports run times with
#' in-sample and out-of-sample mean squared errors. It is meant to give a
#' feel for the backends on a given series, not as a benchmark.
#'
#' Every configuration is fitted from the same random seed and with the same
#' arguments in \code{...}, so the specification of the network (lags,
#' differencing, seasonal dummies, hidden nodes) is the same and only the
#' training differs. Configurations that need torch are skipped with a message
#' if torch is not installed.
#'
#' @param y Time series to use.
#' @param h Number of observations held out and forecast. Defaults to the
#'   longest seasonal period of \code{y}, or 10 for non-seasonal series.
#' @param reps Number of networks in each ensemble.
#' @param configs Named list of configurations, each a list of arguments for
#'   \code{\link{mlp}}, e.g. \code{list(backend = "torch", act.fct = "tanh")}.
#'   If \code{NULL}, compares neuralnet, torch with neuralnet's settings, and
#'   torch with the settings suggested in \code{\link{mlp}}.
#' @param seed Random seed used for every configuration.
#' @param ... For \code{compare_backends}, further arguments passed to \code{\link{mlp}} for every configuration. For \code{plot}, arguments passed to \code{\link[graphics]{plot}} (e.g. \code{xlim}, \code{main}); by default the plot shows the hold-out and the 3\code{h} observations before it. Unused by \code{print}.
#'
#' @return An object of class \code{nnfor_comparison}: a list with
#'   \code{results} (a data frame with one row per configuration: fit and
#'   forecast time in seconds, in-sample and out-of-sample MSE), the
#'   \code{forecasts}, the \code{fits}, and the series split into
#'   \code{train} and \code{test}. Use \code{plot} to see the forecasts.
#' @seealso \code{\link{mlp}}, \code{\link{nnfor_map}}.
#' @examples
#' # Runs with neuralnet only, so it does not need torch
#' cmp <- compare_backends(AirPassengers, reps = 2, configs = list(
#'   "neuralnet: logistic" = list(backend = "neuralnet"),
#'   "neuralnet: tanh"     = list(backend = "neuralnet", act.fct = "tanh")))
#' print(cmp)
#' plot(cmp)
#'
#' \dontrun{
#'  cmp <- compare_backends(AirPassengers)
#'  print(cmp)
#'  plot(cmp)
#'  # Own configurations
#'  compare_backends(AirPassengers, configs = list(
#'    "torch, tanh" = list(backend = "torch", act.fct = "tanh"),
#'    "torch, relu" = list(backend = "torch", act.fct = "relu", algorithm = "adam",
#'                         learningrate = 0.01, stepmax = 2000)))
#' }
#'
#' \dontrun{
#'  # Stress test: many backend configurations with large ensembles. This takes a
#'  # while; neuralnet's time grows with reps (training it in parallel helps),
#'  # while torch trains all members of an ensemble in one batch.
#'  options(nnfor.cores = max(1, parallel::detectCores() - 1))  # neuralnet only
#'  adam <- list(algorithm = "adam", learningrate = 0.01, stepmax = 3000)
#'  configs <- list(
#'    "neuralnet: logistic, rprop+" = list(backend = "neuralnet"),
#'    "neuralnet: tanh, rprop+"     = list(backend = "neuralnet", act.fct = "tanh"),
#'    "torch: logistic, rprop"      = list(backend = "torch"),
#'    "torch: tanh, rprop"          = list(backend = "torch", act.fct = "tanh"),
#'    "torch: gelu, adam"           = c(list(backend = "torch", act.fct = "gelu"), adam),
#'    "torch: silu, adam"           = c(list(backend = "torch", act.fct = "silu"), adam),
#'    "torch: relu, adam"           = c(list(backend = "torch", act.fct = "relu"), adam),
#'    "torch: leaky_relu, adam"     = c(list(backend = "torch", act.fct = "leaky_relu"), adam),
#'    "torch: mish, adamw"          = list(backend = "torch", act.fct = "mish", algorithm = "adamw",
#'                                         learningrate = 0.01, stepmax = 3000),
#'    "torch: elu, rmsprop"         = list(backend = "torch", act.fct = "elu", algorithm = "rmsprop",
#'                                         learningrate = 0.001, stepmax = 3000),
#'    "torch: softplus, adagrad"    = list(backend = "torch", act.fct = "softplus",
#'                                         algorithm = "adagrad", learningrate = 0.05,
#'                                         stepmax = 3000),
#'    "torch: tanh, sgd"            = list(backend = "torch", act.fct = "tanh", algorithm = "sgd",
#'                                         learningrate = 0.001, stepmax = 5000))
#'  cmp <- compare_backends(AirPassengers, reps = 100, configs = configs)
#'  print(cmp)
#'  plot(cmp)
#'  # Configurations ranked by out-of-sample MSE
#'  cmp$results[order(cmp$results$MSE.out), c("configuration", "fit.time", "MSE.in", "MSE.out")]
#' }
#'
#' @export
compare_backends <- function(y = datasets::AirPassengers, h = NULL, reps = 10, configs = NULL,
                             seed = 1, ...) {
  check_ts(y)
  ff <- get_frequencies(y)
  h <- h %||% if (max(ff) > 1) max(ff) else 10
  n <- length(y)
  if (h >= n - 2 * max(ff)) stop("Series too short for a hold-out of h observations.", call. = FALSE)
  train <- stats::window(y, end = time(y)[n - h])
  test <- stats::window(y, start = time(y)[n - h + 1])

  configs <- configs %||% default_backend_configs()
  if (is.null(names(configs)) || any(!nzchar(names(configs)))) {
    stop("configs must be a named list.", call. = FALSE)
  }
  needs.torch <- vapply(configs, function(cfg) identical(cfg$backend, "torch"), logical(1))
  if (any(needs.torch) && !torch_available()) {
    message("torch is not installed; skipping: ", paste(names(configs)[needs.torch], collapse = ", "))
    configs <- configs[!needs.torch]
  }

  runs <- lapply(names(configs), function(nm) {
    set.seed(seed)
    args <- c(list(y = train, reps = reps), configs[[nm]], list(...))
    t.fit <- system.time(fit <- do.call(mlp, args))[["elapsed"]]
    t.frc <- system.time(frc <- forecast(fit, h = h))[["elapsed"]]
    list(fit = fit, forecast = frc$mean,
         row = data.frame(configuration = nm, backend = fit$backend,
                          fit.time = t.fit, forecast.time = t.frc, MSE.in = fit$MSE,
                          MSE.out = mean((as.numeric(test) - as.numeric(frc$mean))^2),
                          stringsAsFactors = FALSE))
  })

  structure(list(results = do.call(rbind, lapply(runs, `[[`, "row")),
                 forecasts = setNames(lapply(runs, `[[`, "forecast"), names(configs)),
                 fits = setNames(lapply(runs, `[[`, "fit"), names(configs)),
                 train = train, test = test, h = h, reps = reps),
            class = "nnfor_comparison")
}

torch_available <- function() {
  requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed()
}

# Default comparison: neuralnet, torch replicating neuralnet, and torch with
# the settings suggested in ?mlp.
default_backend_configs <- function() {
  list("neuralnet" = list(backend = "neuralnet"),
       "torch (neuralnet settings)" = list(backend = "torch"),
       "torch (suggested settings)" = c(list(backend = "torch"), torch_suggested_settings()))
}

# Settings suggested in ?mlp for the torch backend.
torch_suggested_settings <- function() {
  list(act.fct = "gelu", algorithm = "adam", learningrate = 0.01, stepmax = 3000)
}

#' @rdname compare_backends
#' @param x Object of class \code{nnfor_comparison}.
#' @param digits Number of significant digits to print.
#' @export
print.nnfor_comparison <- function(x, digits = 3, ...) {
  writeLines(paste0("Backend comparison: ", x$reps, " networks per ensemble, last ", x$h,
                    " observations held out.\n"))
  res <- x$results
  num <- vapply(res, is.numeric, logical(1))
  res[num] <- lapply(res[num], signif, digits = digits)
  names(res) <- c("Configuration", "Backend", "Fit (s)", "Forecast (s)", "MSE in-sample",
                  "MSE out-of-sample")
  print(res, row.names = FALSE)
  invisible(x)
}

#' @rdname compare_backends
#' @param col Colours of the forecasts. Recycled if there are more
#'   configurations than colours, with a new line type each round so that
#'   every configuration stays distinguishable.
#' @param lty Line types of the forecasts. By default solid for the first
#'   round of colours, dashed for the second, and so on.
#' @param legend Logical, add a legend.
#' @export
plot.nnfor_comparison <- function(x, col = NULL, lty = NULL, legend = TRUE, ...) {
  k <- length(x$forecasts)
  col <- col %||% c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
  lty <- rep_len(lty %||% ((seq_len(k) - 1) %/% length(col) + 1), k)
  col <- rep_len(col, k)
  y <- ts(c(x$train, x$test), start = start(x$train), frequency = frequency(x$train))
  # By default show the hold-out and the three seasons (or 3h observations) before it
  user <- list(...)
  first <- max(1, length(x$train) - 3 * x$h + 1)
  xl <- user[["xlim"]] %||% c(time(y)[first], max(time(y)))
  yl <- visible_ylim(list(y, ts(do.call(cbind, x$forecasts), start = start(x$test),
                                frequency = frequency(y))), xl)
  do.call(plot, merge_args(list(x = y, xlim = xl, ylim = yl, xlab = "", ylab = "",
                                main = "Out-of-sample forecasts by backend"), user))
  abline(v = time(x$test)[1] - deltat(y) / 2, lty = 3, col = "grey")
  for (i in seq_len(k)) lines(x$forecasts[[i]], col = col[i], lty = lty[i], lwd = 2)
  if (legend) {
    legend("topleft", legend = names(x$forecasts), col = col, lty = lty, lwd = 2, bty = "n",
           cex = 0.8, text.col = par("fg"))
  }
  invisible(x)
}
