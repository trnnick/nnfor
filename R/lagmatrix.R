#' Create matrix with lead/lags of an input vector.
#'
#' Create matrix with lead/lags of an input vector.
#'
#' @param x Input variable.
#' @param lag Vector of leads and lags. Positive numbers are lags, negative are leads. O is the original x.
#'
#' @return
#'   \code{lmat} - Resulting matrix.
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @examples
#'   x <- rnorm(10)
#'   lagmatrix(x,c(0,1,-1))
#' @export lagmatrix
#'

lagmatrix <- function(x,lag){

    # Construct matrix with lead and lags
    n <- length(x)
    k <- length(lag)
    # How much to expand for leads and lags
    mlg <- max(c(0,lag[lag>0]))
    mld <- max(abs(c(0,lag[lag<0])))
    # Assign values
    lmat <- array(NA,c(n+mlg+mld,k))
    for (i in 1:k){
        lmat[(1+lag[i]+mld):(n+lag[i]+mld),i] <- x
    }
    # Trim lmat for expansion
    lmat <- lmat[(mld+1):(mld+n),,drop=FALSE]
    return(lmat)
}
