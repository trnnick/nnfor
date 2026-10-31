#' @title ELM (fast) neural network.
#' @description Fit ELM (fast) neural network. This is an ELM implementation that does not rely on neuralnets package.
#'
#' @param y Target variable.
#' @param x Explanatory variables. Each column is a variable.
#' @param hd Starting number of hidden nodes (scalar). Use NULL to automatically specify.
#' @param type Estimation type for output layer weights. Can be "lasso" (lasso with CV), "ridge" (ridge regression with CV), "step" (stepwise regression with AIC) or "ls" (linear regression; "lm" is accepted as a synonym).
#' @param reps Number of networks to train.
#' @param comb Combination operator for forecasts when reps > 1. Can be "median", "mode" (based on KDE estimation) and "mean".
#' @param direct Use direct input-output connections to model strictly linear effects. Can be TRUE or FALSE.
#' @param linscale Scale inputs linearly between -0.8 to 0.8. If output == "logistic" then scaling is between 0 and 1.
#' @param output Type of output layer. It can be "linear" or "logistic". If "logistic" then type must be set to "lasso". Logistic outputs are probabilities from the inverse logit of the output layer and, if \code{linscale = TRUE}, back to the range of \code{y}.
#' @param core If TRUE skips calculation of final fitted values and MSE. Called internally by \code{"elm"} function.
#' @param ortho If TRUE then the initial weights between the input and hidden layers are orthogonal (only when number of input variable <= sample size).
#'
#' @return An object of class "\code{elm.fast}".
#'   The function \code{plot} produces a plot the network fit (see \code{\link{plot.elm.fast}}).
#'   An object of class \code{"elm.fast"} is a list containing the following elements:
#' \itemize{
#' \item \code{hd}: Number of hidden nodes. This is a vector with a different number for each training repetition.
#' \item \code{W.in}: Input weights for each training repetition.
#' \item \code{W}: Output layer weights for each repetition.
#' \item \code{b}: Output node bias for each training repetition.
#' \item \code{W.dct}: Direct connection weights argument if direct == TRUE for each training repetition. Otherwise NULL.
#' \item \code{fitted.all}: Fitted values for each training repetition.
#' \item \code{fitted}: Ensemble fitted values.
#' \item \code{y}: Target variable.
#' \item \code{type}: Estimation used for output layer weights.
#' \item \code{comb}: Combination operator used.
#' \item \code{direct}: Presence of direct input-output connections.
#' \item \code{minmax}: If scaling is used this contains the scaling information for the target variable.
#' \item \code{minmax.x}: If scaling is used this contains the scaling information for the input variables.
#' \item \code{MSE}: In-sample Mean Squared Error.
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}.
#' @references
#' \itemize{
#' \item For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.
#' \item For ELMs see: Huang G.B., Zhou H., Ding X. (2006) Extreme learning machine: theory and applications. \emph{Neurocomputing}, \bold{70}(\bold{1}), 489-501.
#' }
#' @keywords mlp thief ts
#' @note This implementation of ELM is more appropriate when the number of inputs is several hundreds. For time series modelling use \code{\link{elm}} instead.
#' @examples
#' \dontshow{
#'  p <- 2000
#'  n <- 150
#'  X <- matrix(rnorm(p*n),nrow=n)
#'  b <- cbind(rnorm(p))
#'  Y <- X %*% b
#'  fit <- elm.fast(Y,X,reps=1)
#'  print(fit)
#' }
#' \dontrun{
#'  p <- 2000
#'  n <- 150
#'  X <- matrix(rnorm(p*n),nrow=n)
#'  b <- cbind(rnorm(p))
#'  Y <- X %*% b
#'  fit <- elm.fast(Y,X)
#'  print(fit)
#' }
#'
#' @export elm.fast

elm.fast <- function(y, x, hd = NULL, type = c("lasso", "ridge", "step", "ls"), reps = 20,
                     comb = c("median", "mean", "mode"), direct = FALSE,
                     linscale = TRUE, output = c("linear", "logistic"),
                     core = FALSE, ortho = FALSE) {

  type <- match.arg(type[1], c("lasso", "ridge", "step", "ls", "lm"))
  comb <- match.arg(comb)
  output <- match.arg(output)
  direct <- as_flag(direct, "direct")
  core <- as_flag(as.logical(core), "core")  # historically given as a string
  linscale <- as_flag(linscale, "linscale")
  ortho <- as_flag(ortho, "ortho")

  if (output == "logistic" && type != "lasso") {
    warning("Logistic output can only be used with lasso, switching to lasso.")
    type <- "lasso"
  }

  x <- as.matrix(x)
  if (length(y) != nrow(x)) {
    stop("Number of fitting sample and input observations do not match", call. = FALSE)
  }
  p <- ncol(x)
  x.names <- colnames(x)

  minmax.y <- minmax.x <- NULL
  y.sc <- y
  if (linscale) {
    sc.y <- linscale(y, minmax = if (output == "logistic") list(mn = 0, mx = 1) else nnfor_scale_range())
    y.sc <- sc.y$x
    minmax.y <- sc.y$minmax
    sc.x <- lapply(seq_len(p), function(j) linscale(x[, j], minmax = nnfor_scale_range()))
    x <- do.call(cbind, lapply(sc.x, `[[`, "x"))
    minmax.x <- lapply(sc.x, `[[`, "minmax")
  }

  if (is.null(hd)) {
    hd <- 100
  }

  # Random numbers first, in the order of the sequential algorithm (input
  # weights, then cross-validation folds, member by member). Members can then
  # run in parallel if options(nnfor.cores) allows, with identical results.
  draws <- lapply(seq_len(reps), function(r) {
    list(w.in = init.w(p, hd), folds = elm_draw_folds(type, length(y)))
  })

  members <- nnfor_lapply(draws, function(d) {
    # Random input weights, orthogonal when possible
    w.in <- d$w.in
    if (ortho && (p + 1) >= hd) {
      w.in <- svd(t(w.in))$v
    }
    if (!is.null(x.names)) {
      rownames(w.in) <- c("Bias", x.names)
    }
    H <- fast.sig(cbind(1, x) %*% w.in)

    # Output layer
    w <- elm_split_weights(elm.train(y.sc, if (direct) cbind(H, x) else H, type, output,
                                     foldid = d$folds),
                           hd, direct, x.names)

    # Eliminate unused neurons
    used <- as.vector(w$W != 0)
    w.in <- w.in[, used, drop = FALSE]
    w.out <- w$W[used, , drop = FALSE]
    list(W.in = w.in, W = w.out, b = w$b, W.dct = w$W.dct,
         fitted = elm_fast_output(elm_fast_raw(x, w.in, w.out, w$b, w$W.dct, direct),
                                  minmax.y, output))
  })

  W.in <- lapply(members, `[[`, "W.in")
  W <- lapply(members, `[[`, "W")
  B <- vapply(members, `[[`, numeric(1), "b")
  Hd <- vapply(W.in, ncol, numeric(1))
  W.dct <- vector("list", reps)
  if (direct) W.dct <- lapply(members, `[[`, "W.dct")
  Y.all <- matrix(vapply(members, function(m) as.vector(m$fitted), numeric(length(y))),
                  nrow = length(y))

  Y.hat <- MSE <- NULL
  if (!core) {
    Y.hat <- if (reps > 1) frc.comb(Y.all, comb) else Y.all
    MSE <- mean((y - Y.hat)^2)
  }

  structure(list("hd" = Hd, "W" = W, "W.in" = W.in, "b" = B, "W.dct" = W.dct,
                 "fitted.all" = Y.all, "fitted" = Y.hat, "y" = y,
                 "type" = type, "comb" = comb, "direct" = direct,
                 "output" = output, "minmax" = minmax.y, "minmax.x" = minmax.x,
                 "MSE" = MSE), class = "elm.fast")
}

# Map raw network output to the target scale. For logistic output the output
# weights are estimated by a binomial glmnet, so its inverse link (the
# logistic function) maps the linear predictor onto the [0, 1] scale used in
# training. As for linear output, the stored target scaling is then reversed.
elm_fast_output <- function(yhat, minmax.y, output) {
  if (output == "logistic") {
    yhat <- stats::plogis(yhat)
  }
  if (!is.null(minmax.y)) linscale(yhat, minmax.y, rev = TRUE)$x else yhat
}

# Initialise layer weights, uniformly in +/- 1/sqrt(p). Row 1 is the bias.
init.w <- function(p, hd) {
  bb <- c(-1, 1) * (1 / sqrt(p))
  matrix(runif((p + 1) * hd, min = bb[1], max = bb[2]), nrow = (p + 1))
}

# Fast sigmoid-like activation x / (1 + |x|).
fast.sig <- function(x) {
  x / (1 + abs(x))
}

#' @rdname predict.elm.fast
#' @title Predictions for ELM (fast) network.
#' @description Calculate predictions for ELM (fast) network.
#'
#' @param object ELM network object, produced using \code{\link{elm.fast}}.
#' @param newx Explanatory variables. Each column is a variable.
#' @param na.rm If TRUE remove columns and object produces an ensemble forecast, then remove any members that give NA in their forecasts.
#' @param ... Unused argument.
#'
#' @return Returns a list with:
#' \itemize{
#' \item \code{Y.hat}: Ensemble prediction.
#' \item \code{Y.all}: Predictions of each training repetition.
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm.fast}}.
#' @examples
#' \dontshow{
#'  p <- 2000
#'  n <- 150
#'  X <- matrix(rnorm(p*n),nrow=n)
#'  b <- cbind(rnorm(p))
#'  Y <- X %*% b
#'  fit <- elm.fast(Y,X,reps=1)
#'  predict(fit,X)
#' }
#' \dontrun{
#'  p <- 2000
#'  n <- 150
#'  X <- matrix(rnorm(p*n),nrow=n)
#'  b <- cbind(rnorm(p))
#'  Y <- X %*% b
#'  fit <- elm.fast(Y,X)
#'  predict(fit,X)
#' }
#'
#' @keywords elm
#' @export
#' @method predict elm.fast

predict.elm.fast <- function(object, newx, na.rm = FALSE, ...) {

  if (!identical(class(object), "elm.fast")) {
    stop("Use exclusively with objects that are of elm.fast class only!", call. = FALSE)
  }
  na.rm <- as_flag(na.rm, "na.rm")
  newx <- as.matrix(newx)

  reps <- length(object$b)
  p <- ncol(newx)
  elm.p <- nrow(object$W.in[[1]]) - 1    # -1 for bias
  if (p != elm.p) {
    stop("newx has incorrect number of variables. ELM trained with ", elm.p, " variables.", call. = FALSE)
  }

  if (!is.null(object$minmax.x)) {
    for (i in 1:p) {
      newx[, i] <- linscale(newx[, i], minmax = object$minmax.x[[i]])$x
    }
  }

  Y.all <- array(NA, c(nrow(newx), reps))
  for (r in 1:reps) {
    Y.all[, r] <- elm_fast_output(
      elm_fast_raw(newx, object$W.in[[r]], object$W[[r]], object$b[r],
                                object$W.dct[[r]], object$direct),
      object$minmax, object$output)
  }

  Y.hat <- if (reps > 1) frc.comb(Y.all, object$comb, na.rm) else Y.all

  list("Y.hat" = Y.hat, "Y.all" = Y.all)
}

# Raw (scaled) output of one elm.fast member.
elm_fast_raw <- function(x, w.in, w.out, b, w.dct, direct) {
  fast.sig(cbind(1, x) %*% w.in) %*% w.out + b + if (direct != TRUE) 0 else x %*% w.dct
}
