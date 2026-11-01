#' Apply minmax linear scaling to a vector
#'
#' Apply minmax linear scaling to a vector.
#'
#' @param x Input vector.
#' @param minmax minmax must be a list with elements "mn", "mx", "mn.orig" and "mx.orig", where "mn" and "mx" refer to the target min and max, and the remaining two refer to the current vector min and max. By default mn=-1 and mx=1. mn.orig and mx.orig can be missing, unless the scaling is reversed.
#' @param rev Reverse scaling back to original: TRUE or FALSE.
#'
#' @return Outputs a list with elements:
#' \itemize{
#' \item \code{x} - Scaled vector.
#' \item \code{minmax} - List with resulting mn, mx, mn.orig and mx.orig. Can be used as input to reverse scaling.
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @examples
#'   y <- rnorm(20)*100
#'   sc <- linscale(y)
#'   x <- sc$x
#'   print(c(min(y),max(y)))
#'   print(c(min(x),max(x)))
#'   sc.rev <- linscale(x,minmax=sc$minmax,rev=TRUE)
#'   print(c(min(sc.rev$x),max(sc.rev$x)))
#'
#' @export linscale

linscale <- function(x, minmax = NULL, rev = FALSE) {
  rev <- isTRUE(rev[1])

  # To reverse the scaling both the scaled and original mn/mx are needed, so
  # that samples with a range different from the one used to infer the
  # scaling can be mapped back.
  if (is.null(minmax)) {
    if (rev) {
      stop("To reverse scaling minmax input is required.", call. = FALSE)
    }
    minmax <- list(mn = -1, mx = 1)
  }
  has <- function(nm) nm %in% names(minmax) && !is.null(minmax[[nm]])

  if (rev) {
    if (!all(vapply(c("mn", "mx", "mn.orig", "mx.orig"), has, logical(1)))) {
      stop("Provided minmax list is not of correct type. It must contain mn, mx, mn.orig and mx.orig to reverse scaling.",
           call. = FALSE)
    }
    mn <- minmax[["mn"]]; mx <- minmax[["mx"]]
    mn.orig <- minmax[["mn.orig"]]; mx.orig <- minmax[["mx.orig"]]
    x.orig <- (mx.orig - mn.orig) * (x - mn) / (mx - mn) + mn.orig
    return(list("x" = x.orig, "minmax" = list("mn" = mn.orig, "mx" = mx.orig, "mn.orig" = mn, "mx.orig" = mx)))
  }

  if (!has("mn") || !has("mx")) {
    stop("Provided minmax list is not of correct type. It must contain mn and mx to apply scaling.",
         call. = FALSE)
  }
  mn <- minmax[["mn"]]; mx <- minmax[["mx"]]
  # Reuse a previously inferred original range if given, so that the scale of
  # a variable stays fixed; otherwise infer it from x.
  if (has("mn.orig") && has("mx.orig")) {
    mn.orig <- minmax[["mn.orig"]]; mx.orig <- minmax[["mx.orig"]]
  } else {
    mn.orig <- min(x); mx.orig <- max(x)
  }
  x.sc <- (mx - mn) * (x - mn.orig) / (mx.orig - mn.orig) + mn
  list("x" = x.sc, "minmax" = list("mn" = mn, "mx" = mx, "mn.orig" = mn.orig, "mx.orig" = mx.orig))
}
