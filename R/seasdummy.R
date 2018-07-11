#' Create seasonal dummy variables.
#'
#' Create binary or trigonometric seasonal dummies.
#'
#' @param n Number of observations to create.
#' @param m Seasonal periodicity. If NULL it will take the information from the provided time series (y argument). See notes.
#' @param y This is an optional time series input that can be used to get seasonal periodicity (m) and the start point.
#' @param type Type of seasonal dummies to create. Can be "bin" for binary and "trg" for trigonometric. See notes.
#' @param full If full is TRUE then keeps the m-th dummy that is co-linear to the rest. See notes.
#'
#' @note If the seasonal periodicity is fractional then the the type will be overriden to trigonometric and only two seasonal dummies with be produced. One cosine and one sine.
#'
#' @return
#'   \code{x} - Array with seasonal dummies.
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @examples
#'   x <- seasdummy(24,12)
#'
#' @export seasdummy

seasdummy <- function(n,m=NULL,y=NULL,type=c("bin","trg"),full=c(FALSE,TRUE)){
# Create deterministic seasonality dummies

    # Default
    type <- match.arg(type,c("bin","trg"))
    full <- full[1]

    # Get time series information if available
    if (!is.null(y)){
      if (is.null(m)){
        m <- frequency(y)
      }
      start <- start(y)
      # Deal with fractional seaosnalities
      isdd <-length(start)==2
      if (isdd){
        start <- start[2]
      } else {
        start <- start %% 1
      }
    } else {
      start <- 1
      isdd <- TRUE
    }

    if (start >= n){
      n.sim <- n + start
    } else {
      n.sim <- n
    }

    if (is.null(m)){
      stop("Seasonality not specified.")
    }

    # Create dummies
    if ((m %% 1) == 0){
      if (type == "bin"){
          x <- matrix(rep(diag(rep(1,m)),ceiling(n.sim/m)),ncol=m,byrow=TRUE)[1:n.sim,,drop=FALSE]
      } else { # trg
          x <- array(0,c(n.sim,m))
          t <- 1:n.sim
          for (i in 1:(m/2)){
              x[,1+(i-1)*2] <- cos((2*t*pi*i)/m+(2*pi*(start-isdd))/m) # Added phase shift for start
              x[,2+(i-1)*2] <- sin((2*t*pi*i)/m+(2*pi*(start-isdd))/m)
          }
      }
      # Trim co-linear dummy
      if (full==FALSE){
          x <- x[,1:(m-1),drop=FALSE]
      }
    } else {
      # Fractional seasonality
      x <- array(0,c(n.sim,2))
      t <- 1:n.sim
      x[,1] <- cos((2*t*pi)/m+(2*pi*(start-isdd))) # Added phase shift for start
      x[,2] <- sin((2*t*pi)/m+(2*pi*(start-isdd)))
    }

    # Shift for start
    if (start > 1 & type == "bin"){ # For type=="trg" it is handled with a phase shift
        x <- rbind(x[start:n, ,drop=FALSE], x[1:(start - 1), ,drop=FALSE])
    }

    # If n.sim is larger, just retain n observations
    x <- x[1:n,,drop=FALSE]

    return(x)

}
