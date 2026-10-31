# Parallel execution of independent ensemble members.
#
# Members are trained in parallel only after every random number they need
# has been drawn, in the order the sequential code would draw them (see
# backend_neuralnet(), elm_fit_output() and elm.fast()). Workers therefore use
# no random numbers and results are identical for any number of cores.
#
# Controlled by options(nnfor.cores = ...):
#   1 (default)        sequential
#   an integer > 1     forked workers (parallel::mclapply) on Linux and macOS;
#                      a temporary socket cluster on Windows
#   a cluster object   parallel::parLapply on that cluster (e.g. from
#                      parallel::makeCluster()). Best on Windows, where starting
#                      a cluster is slow and is then only paid once.

nnfor_cores <- function() {
  cores <- getOption("nnfor.cores", 1L)
  if (inherits(cores, "cluster")) return(cores)
  if (!is.numeric(cores) || length(cores) != 1 || is.na(cores) || cores < 1) {
    stop("Option nnfor.cores must be a positive integer or a cluster.", call. = FALSE)
  }
  as.integer(cores)
}

# lapply() that may run in parallel. Warnings raised in workers are re-issued
# in the calling session; errors are propagated.
nnfor_lapply <- function(X, FUN) {
  cores <- nnfor_cores()
  if (!inherits(cores, "cluster") && (cores == 1 || length(X) < 2)) {
    return(lapply(X, FUN))
  }

  run <- function(x) {
    warn <- character(0)
    value <- withCallingHandlers(FUN(x), warning = function(w) {
      warn <<- c(warn, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
    list(value = value, warnings = warn)
  }

  if (inherits(cores, "cluster")) {
    res <- parallel::parLapply(cores, X, run)
  } else if (.Platform$OS.type == "windows") {
    cl <- parallel::makeCluster(min(cores, length(X)))
    on.exit(parallel::stopCluster(cl))
    res <- parallel::parLapply(cl, X, run)
  } else {
    res <- parallel::mclapply(X, run, mc.cores = min(cores, length(X)))
    failed <- vapply(res, inherits, logical(1), what = "try-error")
    if (any(failed)) {
      stop(attr(res[[which(failed)[1]]], "condition"))
    }
  }

  for (w in unique(unlist(lapply(res, `[[`, "warnings")))) warning(w, call. = FALSE)
  lapply(res, `[[`, "value")
}
