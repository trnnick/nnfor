# Regression harness: run the same seeded cases against an nnfor build and
# save everything numeric that a user could observe.
# Usage: Rscript cases.R <lib_path> <out.rds>
args <- commandArgs(TRUE)
.libPaths(c(args[1], .libPaths()))
suppressMessages(library(nnfor))
suppressMessages(library(forecast))
pdf(NULL)

# Parallel mode: NNFOR_CORES = 1 (default), an integer, or "psock" for a
# 2-worker socket cluster (the Windows route).
cores <- Sys.getenv("NNFOR_CORES", "1")
if (cores == "psock") {
  cl <- parallel::makeCluster(2)
  parallel::clusterCall(cl, function(p) .libPaths(c(p, .libPaths())), args[1])
  options(nnfor.cores = cl)
} else {
  options(nnfor.cores = as.integer(cores))
}

grab <- function(fit, frc) {
  list(fitted = as.numeric(fit$fitted), MSE = fit$MSE, lags = fit$lags,
       xreg.lags = fit$xreg.lags, difforder = fit$difforder, sdummy = fit$sdummy,
       ff.det = fit$ff.det, det.type = fit$det.type, hd = fit$hd, MSEH = fit$MSEH,
       mean = as.numeric(frc$mean), all = unclass(frc$all.mean),
       tsp.f = tsp(fit$fitted), tsp.m = tsp(frc$mean))
}

set.seed(1); n <- length(AirPassengers)
X <- cbind(rnorm(n + 24), cumsum(rnorm(n + 24)))
yq <- ts(100 + cumsum(rnorm(80)) + rep(c(5, -3, 2, -4), 20), frequency = 4, start = c(2000, 1))
ym <- forecast::msts(200 + 10 * sin(2 * pi * (1:240) / 4) + 20 * sin(2 * pi * (1:240) / 12) +
                     rnorm(240), seasonal.periods = c(4, 12))
yn <- ts(as.numeric(Nile))

cases <- list(
  mlp_auto      = function() { f <- mlp(AirPassengers, reps = 2); grab(f, forecast(f, h = 12)) },
  mlp_manual    = function() { f <- mlp(AirPassengers, reps = 2, hd = c(3, 2), difforder = 0, sel.lag = FALSE); grab(f, forecast(f, h = 6)) },
  mlp_xreg      = function() { f <- mlp(AirPassengers, reps = 2, xreg = X, xreg.lags = list(0:2, 1)); grab(f, forecast(f, h = 12, xreg = X)) },
  mlp_xreg_auto = function() { f <- mlp(AirPassengers, reps = 1, xreg = X); grab(f, forecast(f, h = 12, xreg = X)) },
  mlp_valid     = function() { f <- mlp(yq, reps = 2, hd.auto.type = "valid", hd.max = 3); grab(f, forecast(f, h = 8)) },
  mlp_cv        = function() { f <- mlp(yq, reps = 2, hd.auto.type = "cv", hd.max = 2); grab(f, forecast(f, h = 8)) },
  mlp_hdelm     = function() { f <- mlp(yq, reps = 1, hd.auto.type = "elm"); grab(f, forecast(f, h = 8)) },
  mlp_mode      = function() { f <- mlp(yq, reps = 3, comb = "mode", difforder = c(1, 4)); grab(f, forecast(f, h = 8)) },
  mlp_trg       = function() { f <- mlp(AirPassengers, reps = 1, det.type = "trg", difforder = 1); grab(f, forecast(f, h = 12)) },
  mlp_keep      = function() { f <- mlp(yq, reps = 1, lags = 1:6, keep = c(F, F, F, F, F, T)); grab(f, forecast(f, h = 4)) },
  mlp_msts      = function() { f <- mlp(ym, reps = 1, difforder = 0); grab(f, forecast(f, h = 12)) },
  mlp_nonseas   = function() { f <- mlp(yn, reps = 2); grab(f, forecast(f, h = 5)) },
  mlp_xonly     = function() { f <- mlp(AirPassengers, reps = 1, lags = 0, xreg = X, xreg.lags = list(1:2, 0:1), sel.lag = FALSE); grab(f, forecast(f, h = 12, xreg = X)) },
  mlp_reuse     = function() { f <- mlp(window(AirPassengers, end = c(1958, 12)), reps = 2)
                               g <- mlp(AirPassengers, model = f); grab(g, forecast(g, h = 12)) },
  mlp_retrain   = function() { f <- mlp(window(AirPassengers, end = c(1958, 12)), reps = 2)
                               g <- mlp(AirPassengers, model = f, retrain = TRUE); grab(g, forecast(g, h = 12)) },
  elm_auto      = function() { f <- elm(AirPassengers, reps = 3); grab(f, forecast(f, h = 12)) },
  elm_step_dir  = function() { f <- elm(yq, reps = 2, type = "step", direct = TRUE); grab(f, forecast(f, h = 8)) },
  elm_lm        = function() { f <- elm(yq, reps = 2, type = "lm", hd = 6); grab(f, forecast(f, h = 8)) },
  elm_fast      = function() { f <- elm(AirPassengers, reps = 3, barebone = TRUE); grab(f, forecast(f, h = 12)) },
  elm_fast_dir  = function() { f <- elm(AirPassengers, reps = 2, barebone = TRUE, direct = TRUE, type = "ridge"); grab(f, forecast(f, h = 12)) },
  elm_xreg      = function() { f <- elm(AirPassengers, reps = 2, xreg = X); grab(f, forecast(f, h = 12, xreg = X)) },
  elm_reuse     = function() { f <- elm(window(AirPassengers, end = c(1958, 12)), reps = 2)
                               g <- elm(AirPassengers, model = f); grab(g, forecast(g, h = 12)) },
  elm_fast_reuse= function() { f <- elm(window(AirPassengers, end = c(1958, 12)), reps = 2, barebone = TRUE)
                               g <- elm(AirPassengers, model = f); grab(g, forecast(g, h = 12)) },
  elm_fast_raw  = function() { Xr <- matrix(rnorm(100 * 30), 100); Yr <- Xr %*% rnorm(30)
                               f <- elm.fast(Yr, Xr, reps = 3, hd = 20)
                               list(fitted = f$fitted, MSE = f$MSE, hd = f$hd,
                                    pred = predict(f, Xr[1:10, ])$Y.hat) },
  lasso_short   = function() { y <- ts(cumsum(rnorm(16)) + 50, frequency = 4)
                               f <- mlp(y, reps = 1, lags = 1:8, difforder = 0); grab(f, forecast(f, h = 4)) },
  lags_dropped  = function() { set.seed(6); n <- 96; x <- rnorm(n + 12)
                               y <- ts(100 + rep(c(10,-5,3,-8,6,0,-2,4,-6,9,-3,-8), 8) + 8 * x[1:n] + rnorm(n, sd = 0.5), frequency = 12)
                               f <- mlp(y, reps = 1, xreg = cbind(x), xreg.lags = list(0:2), difforder = 0)
                               grab(f, forecast(f, h = 12, xreg = cbind(x))) },
  msts_bin      = function() { f <- mlp(ym, reps = 1, difforder = 0, det.type = "bin"); grab(f, forecast(f, h = 12)) },
  elm_multilayer= function() { f <- elm(yq, reps = 2, hd = c(6, 4)); grab(f, forecast(f, h = 8)) },
  elm_fast_logit= function() { Xr <- matrix(rnorm(120 * 4), 120); Yr <- as.numeric(Xr %*% c(1, -1, 0.5, 0) + rnorm(120) > 0)
                               f <- elm.fast(Yr, Xr, reps = 3, hd = 15, output = "logistic")
                               list(fitted = f$fitted, hd = f$hd, pred = predict(f, Xr[1:10, ])$Y.hat,
                                    pred1 = predict(f, Xr[1, , drop = FALSE])$Y.hat) },
  elm_fast_dirr = function() { Xr <- matrix(rnorm(100 * 5), 100); Yr <- Xr %*% rnorm(5)
                               f <- elm.fast(Yr, Xr, reps = 2, hd = 10, direct = TRUE, type = "step")
                               list(fitted = f$fitted, MSE = f$MSE, hd = f$hd,
                                    pred = predict(f, Xr[1:10, ])$Y.hat) }
)

out <- lapply(names(cases), function(nm) {
  set.seed(42)
  res <- tryCatch(suppressWarnings(cases[[nm]]()), error = function(e) paste("ERROR:", conditionMessage(e)))
  cat(sprintf("%-15s %s\n", nm, if (is.character(res)) res else "ok"))
  res
})
names(out) <- names(cases)
saveRDS(out, args[2])
