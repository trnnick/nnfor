#' @export
#' @method print mlp
print.mlp <- function(x, ...) {
  print_network(x)
  invisible(x)
}

#' @export
#' @method print elm
print.elm <- function(x, ...) {
  print_network(x)
  invisible(x)
}

#' @export
#' @method print elm.fast
print.elm.fast <- function(x, ...) {
  reps <- length(x$b)
  writeLines(paste0("ELM (fast) with fit with ", min(x$hd), " up to ", max(x$hd), " hidden node",
                    plural(any(x$hd > 1)), direct_text(x), " and ", reps, " repetition",
                    plural(reps > 1), "."))
  if (reps > 1) {
    writeLines(paste0("Forecast combined using the ", x$comb, " operator."))
  }
  writeLines(paste0("Output weight estimation using: ", x$type, "."))
  if (!is.null(x$MSE)) {
    writeLines(paste0("MSE: ", round(x$MSE, 4), "."))
  }
  invisible(x)
}

plural <- function(cond) if (cond) "s" else ""

direct_text <- function(x) if (isTRUE(x$direct)) ", direct output connections" else ""

print_network <- function(x) {
  is.elm <- inherits(x, "elm")
  is.fast <- inherits(x, "elm.fast")
  reps <- n_members(x)
  hd <- x$hd

  hdt <- if (is.fast) {
    paste0(min(hd), " up to ", max(hd))
  } else if (length(hd) > 1) {
    paste0("(", paste0(hd, collapse = ","), ")")
  } else {
    hd
  }

  writeLines(paste0(if (is.elm) "ELM" else "MLP", if (is.fast) " (fast)" else "",
                    " fit with ", hdt, " hidden node", plural(any(hd > 1)),
                    if (is.elm) direct_text(x) else "",
                    " and ", reps, " repetition", plural(reps > 1), "."))
  if (length(x$difforder) > 0) {
    writeLines(paste0("Series modelled in differences: ", paste0("D", x$difforder, collapse = ""), "."))
  }
  if (has_lags(x$lags)) {
    writeLines(paste0("Univariate lags: (", paste0(x$lags, collapse = ","), ")"))
  }
  used <- which(lengths(x$xreg.lags) > 0)
  if (length(used) > 0) {
    writeLines(paste0(length(used), " regressor", plural(length(used) > 1), " included."))
    for (i in seq_along(used)) {
      writeLines(paste0("- Regressor ", i, " lags: (", paste0(x$xreg.lags[[used[i]]], collapse = ","), ")"))
    }
  }
  if (isTRUE(x$sdummy)) {
    writeLines("Deterministic seasonal dummies included.")
  }
  if (reps > 1) {
    writeLines(paste0("Forecast combined using the ", x$comb, " operator."))
  }
  if (is.elm) {
    writeLines(paste0("Output weight estimation using: ", x$type, "."))
  }
  writeLines(paste0("MSE: ", round(x$MSE, 4), "."))
}
