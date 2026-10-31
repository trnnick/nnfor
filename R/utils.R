# Internal helpers shared across the package.
#
# Naming convention: exported functions keep their historical dot.case names
# (mlp, elm.fast, linscale, ...). Internal helpers use snake_case so they can
# never be mistaken for S3 methods (e.g. the old `plot.net` would have been
# dispatched for an object of class "net").

# Range that targets and inputs are linearly scaled to before training.
nnfor_scale_range <- function() list(mn = -0.8, mx = 0.8)

`%||%` <- function(a, b) if (is.null(a)) b else a

# Historically logical arguments were declared as c(FALSE, TRUE) and the
# first element used. Accept both that and a plain logical.
as_flag <- function(x, name) {
  x <- x[1]
  if (!is.logical(x) || is.na(x)) {
    stop("Argument `", name, "' must be TRUE or FALSE.", call. = FALSE)
  }
  x
}

check_ts <- function(y) {
  if (!inherits(y, "ts")) {
    stop("Input y must be of class ts or msts.", call. = FALSE)
  }
  invisible(y)
}

# Seasonal periods of a ts or msts object.
get_frequencies <- function(y) {
  if (inherits(y, "msts")) attr(y, "msts") else frequency(y)
}

# Seasonal periods to model: `m` if given, otherwise those of `y`.
resolve_periods <- function(m, y) {
  if (is.null(m)) return(get_frequencies(y))
  if (!is.numeric(m) || length(m) == 0 || anyNA(m) || any(m < 1)) {
    stop("Argument `m' must be NULL or a vector of seasonal periods >= 1.", call. = FALSE)
  }
  m
}

# Number of univariate lags actually requested. `lags = 0` means none.
has_lags <- function(lags) length(lags) > 0 && !all(lags == 0)

n_xreg <- function(xreg) if (is.null(xreg)) 0L else NCOL(xreg)

check_xreg <- function(xreg, xreg.lags, y) {
  if (is.null(xreg)) return(NULL)
  xreg <- as.matrix(xreg)
  if (!is.null(xreg.lags) && length(xreg.lags) != ncol(xreg)) {
    stop("Argument xreg.lags must be a list with as many elements as xreg variables (columns).",
         call. = FALSE)
  }
  if (nrow(xreg) < length(y)) {
    stop("xreg must have at least as many rows as y.", call. = FALSE)
  }
  xreg
}

check_hd <- function(hd) {
  if (!is.null(hd) && (!is.numeric(hd) || any(hd < 1) || any(hd %% 1 != 0))) {
    stop("Argument `hd' must be NULL or a vector of positive integers.", call. = FALSE)
  }
  hd
}

# Start of the forecast period. Temporal aggregation can mess up the
# start/end attributes of a ts, so compute it from end() and deltat().
forecast_start <- function(y) {
  fend <- end(y)
  if (length(fend) == 1) {
    fend + deltat(y)
  } else {
    fend[1] + (fend[2] - 1) / frequency(y) + deltat(y)
  }
}

# Merge user-supplied graphical arguments over package defaults. Anything the
# user passes wins, so every default is overridable.
merge_args <- function(defaults, user) {
  utils::modifyList(defaults, user, keep.null = FALSE)
}

# cbind a (possibly empty or NULL) list of matrices; NULL if nothing to bind.
cbind_list <- function(l) {
  l <- Filter(Negate(is.null), l)
  if (length(l)) do.call(cbind, l) else NULL
}
