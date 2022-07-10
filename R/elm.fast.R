#' @title ELM (fast) neural network.
#' @description Fit ELM (fast) neural network. This is an ELM implementation that does not rely on neuralnets package.
#'
#' @param y Target variable.
#' @param x Explanatory variables. Each column is a variable.
#' @param hd Starting number of hidden nodes (scalar). Use NULL to automatically specify.
#' @param type Estimation type for output layer weights. Can be "lasso" (lasso with CV), "ridge" (ridge regression with CV), "step" (stepwise regression with AIC) or "lm" (linear regression).
#' @param reps Number of networks to train.
#' @param comb Combination operator for forecasts when reps > 1. Can be "median", "mode" (based on KDE estimation) and "mean".
#' @param direct Use direct input-output connections to model strictly linear effects. Can be TRUE or FALSE.
#' @param linscale Scale inputs linearly between -0.8 to 0.8. If output == "logistic" then scaling is between 0 and 1.
#' @param output Type of output layer. It can be "linear" or "logistic". If "logistic" then type must be set to "lasso".
#' @param core If TRUE skips calculation of final fitted values and MSE. Called internally by \code{"elm"} function.
#' @param ortho If TRUE then the initial weights between the input and hidden layers are orthogonal (only when number of input variable <= sample size).
#'
#' @return An object of class "\code{elm.fast}".
#'   The function \code{plot} produces a plot the network fit.
#'   An object of class \code{"elm.fast"} is a list containing the following elements:
#' \itemize{
#' \item{\code{hd}{ - Number of hidden nodes. This is a vector with a different number for each training repetition.}}
#' \item{\code{W.in}{ - Input weights for each training repetition.}}
#' \item{\code{W}{ - Output layer weights for each repetition.}}
#' \item{\code{b}{ - Output node bias for each training repetition.}}
#' \item{\code{W.dct}{ - Direct connection weights argument if direct == TRUE for each training repetition. Otherwuse NULL.}}
#' \item{\code{fitted.all}{ - Fitted values for each training repetition.}}
#' \item{\code{fitted}{ - Ensemble fitted values.}}
#' \item{\code{y}{ - Target variable.}}
#' \item{\code{type}{ - Estimation used for output layer weights.}}
#' \item{\code{comb}{ - Combination operator used.}}
#' \item{\code{direct}{ - Presence of direct input-output connections.}}
#' \item{\code{minmax}{ - If scaling is used this contains the scaling information for the target variable.}}
#' \item{\code{minmax.x}{ - If scaling is used this contains the scaling information for the input variables.}}
#' \item{\code{MSE}{ - In-sample Mean Squared Error.}}
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}.
#' @references
#' \itemize{
#' \item{For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.}
#' \item{For ELMs see: Huang G.B., Zhou H., Ding X. (2006) Extreme learning machine: theory and applications. \emph{Neurocomputing}, \bold{70}(\bold{1}), 489-501.}
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

elm.fast <- function(y,x,hd=NULL,type=c("lasso","ridge","step","ls"),reps=20,
                     comb=c("median","mean","mode"),direct=c(FALSE,TRUE),
                     linscale=c(TRUE,FALSE),output=c("linear","logistic"),
                     core=c("FALSE","TRUE"),ortho=c(FALSE,TRUE)){

  type <- match.arg(type,c("lasso","ridge","step","ls"))
  comb <- match.arg(comb,c("median","mean","mode"))
  output <- match.arg(output,c("linear","logistic"))
  direct <- direct[1]
  core <- core[1]
  linscale <- linscale[1]
  ortho <- ortho[1]

  if (output == "logistic" & type != "lasso"){
      warning("Logisitc output can only be used with lasso, switching to lasso.")
      type <- "lasso"
  }

  n.y <- length(y)
  x.rc <- dim(x)
  n.x <- x.rc[1]
  p <- x.rc[2]
  x.names <- colnames(x)

  if (linscale){
      # Scale target
      if (output == "logistic"){
          sc.y <- linscale(y,minmax=list("mn"=-0,"mx"=1))
      } else {
          sc.y <- linscale(y,minmax=list("mn"=-.8,"mx"=0.8))
      }
      y.sc <- sc.y$x
      minmax.y <- sc.y$minmax
      # Scale all x's
      x.sc <- apply(x,2,linscale,minmax=list("mn"=-.8,"mx"=0.8))
      minmax.x <- sapply(x.sc, "[", 2)
      x <- do.call(cbind,sapply(x.sc, "[", 1))
  } else {
      y.sc <- y
      sc.y <- sc.x <- NULL
      minmax.x <- minmax.y <- NULL
  }

  if (n.y != n.x){
    stop("Number of fitting sample and input observations do not match")
  }

  if (is.null(hd)){
    hd <- 100
  }

  # Initialise variables to store elm
  W.in <- W <- W.dct <- vector("list",reps)
  B <- Hd <- vector("numeric",reps)
  Y.all <- array(NA,c(n.y,reps))

  # Iterate for each training replication
  for (r in 1:reps){

    # Calculate hidden layer values
    w.in <- init.w(p,hd)
    if (ortho == TRUE){
    # Orthogonal weights (when possible)
      if ((p+1) >= hd){
        w.in <- svd(t(w.in))$v
      }
    }
    if (!is.null(x.names)){
        rownames(w.in) <- c("Bias",x.names)
    }
    hd.hat <- fast.sig(cbind(1,x) %*% w.in)
    # Allow direct connections
    if (direct==TRUE){
      z <- cbind(hd.hat,x)
    } else {
      z <- hd.hat
    }
    # Optimise last layer
    w.out <- elm.train(y.sc,z,type,x,direct,hd,output)

    # Distribute weights
    B[r] <- w.out[1]                                  # Bias (Constant)
    if (direct == TRUE){                              # Direct connections
        w.dct <- w.out[(1+hd+1):(1+hd+p),,drop=FALSE]
        if (!is.null(x.names)){
            rownames(w.dct) <- x.names
        }
        W.dct[[r]] <- w.dct
    }
    w.out <- w.out[2:(1+hd),,drop=FALSE]              # Hidden layer

    # Eliminate unused neurons
    W.in[[r]] <- w.in[,w.out != 0, drop=FALSE]
    Hd[r] <- dim(W.in[[r]])[2]
    W[[r]] <- w.out[w.out != 0,, drop=FALSE]

    # Predict fitted values
    Y.all[,r] <- predict.elm.fast.internal(x,W.in[[r]],W[[r]],B[r],W.dct[[r]],direct)

    # Reverse scaling or apply logistic
    if (linscale){
        if (output != "logistic"){
            Y.all[,r] <- linscale(Y.all[,r],sc.y$minmax,rev=TRUE)$x
        }
    }
    if (output == "logistic"){
        Y.all[,r] <- linscale(fast.sig(Y.all[,r]),minmax=list("mn"=0,"mx"=1))$x
    }

  }

  if (core == FALSE){
      if (reps > 1){
          Y.hat <- frc.comb(Y.all,comb)
      } else {
          Y.hat <- Y.all
      }
      MSE <- mean((y-Y.hat)^2)
  } else {
    Y.hat <- NULL
    MSE <- NULL
  }

  return(structure(list("hd"=Hd,"W"=W,"W.in"=W.in,"b"=B,"W.dct"=W.dct,
                        "fitted.all"=Y.all,"fitted"=Y.hat,"y"=y,
                        "type"=type,"comb"=comb,"direct"=direct,
                        "output"=output,"minmax"=minmax.y,"minmax.x"=minmax.x,
                        "MSE"=MSE),class="elm.fast"))


  # out <- structure(list("net"=NULL,"hd"=f.elm$hd,"W"=f.elm$W,"W.in"=f.elm$W.in,"b"=f.elm$b,"W.dct"=f.elm$Wdct,
  #                       "lags"=lags,"xreg.lags"=xreg.lags,"difforder"=difforder,
  #                       "sdummy"=sdummy,"ff.det"=ff.det,"det.type"=det.type,"y"=y,"minmax"=sc$minmax,"xreg.minmax"=xreg.minmax,
  #                       "comb"=comb,"type"=type,"direct"=direct,"fitted"=yout,"MSE"=MSE),class=c("elm.fast","elm"))

}

init.w <- function(p,hd){
# Initialise layer weights
  bb <- c(-1,1)*(1/sqrt(p))
  w <- matrix(runif((p+1)*hd,min=bb[1],max=bb[2]),nrow=(p+1)) # p + 1 for bias
  return(w)
}

fast.sig <- function(x){
  y <- x/(1+abs(x))
  return(y)
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
#' \item{\code{Y.hat}{ - Ensemble prediction.}}
#' \item{\code{Y.all}{ - Predictions of each training repetition.}}
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

predict.elm.fast <- function(object,newx,na.rm=c(FALSE,TRUE),...){
# Prediction for elm.fast object

    if (any(class(object) != "elm.fast")){
        stop("Use exclusively with objects that are of elm.fast class only!")
    }

    na.rm <- na.rm[1]

    reps <- length(object$b)
    W.in <- object$W.in
    output <- object$output
    minmax.y <- object$minmax
    minmax.x <- object$minmax.x

    n <- dim(newx)[1]
    p <- dim(newx)[2]
    elm.p <- dim(W.in[[1]])[1]-1    # -1 for bias
    if (p != elm.p){
      stop(paste0("newx has incorrect number of variables. ELM trained with ",elm.p,"variables"))
    }

    # Apply scaling to xnew
    if (!is.null(minmax.x)){
      for (i in 1:p){
        newx[,i] <- linscale(newx[,i],minmax=minmax.x[[i]])$x
      }
    }

    Y.all <- array(NA,c(n,reps))
    for (r in 1:reps){
        Y.all[,r] <- predict.elm.fast.internal(newx,W.in[[r]],object$W[[r]],object$b[r],object$W.dct[[r]],object$direct)

        # Reverse scaling or apply logistic
        if (!is.null(minmax.y)){
            if (output != "logistic"){
                Y.all[,r] <- linscale(Y.all[,r],minmax.y,rev=TRUE)$x
            }
        }
        if (output == "logistic"){
            Y.all[,r] <- linscale(fast.sig(Y.all[,r]),minmax=list("mn"=0,"mx"=1))$x
        }

    }

    if (reps > 1){
        Y.hat <- frc.comb(Y.all,object$comb,na.rm)
    } else {
        Y.hat <- Y.all
    }


    return(list("Y.hat"=Y.hat,"Y.all"=Y.all))

}

#' @export
#' @method plot elm.fast

plot.elm.fast <- function(x, ...){

  reps <- length(x$b)
  yy <- range(cbind(x$y,x$fitted,x$fitted.all))
  yy <- xx <- c(min(yy),max(yy)) + c(-1,+1)*0.04*diff(yy)
  plot(NA,NA,xlab="Fitted",ylab="Actual",xlim=xx,ylim=yy)
  lines(xx+c(-1,1)*0.2*diff(xx),yy+c(-1,1)*0.2*diff(yy),col="grey")
  for (r in 1:reps){
    points(x$fitted.all[,r],x$y,col="gray30",pch=".")
  }
  points(x$fitted,x$y,pch=20)

}

#' @export
#' @method print elm.fast

print.elm.fast <- function(x, ...){

    hd <- x$hd
    reps <- length(x$b)

    hdt <- paste0(min(hd)," up to ",max(hd))
    if (any(hd>1)){
        hde <- "s"
    } else {
        hde <- ""
    }

    dtx <- ""
    if (x$direct == TRUE){
        dtx <- ", direct output connections"
    }

    writeLines(paste0("ELM (fast) with"," fit with ", hdt," hidden node", hde, dtx," and ", reps, " repetition",if(reps>1){"s"},"."))
    if (reps>1){
        writeLines(paste0("Forecast combined using the ", x$comb, " operator."))
    }
    writeLines(paste0("Output weight estimation using: ", x$type, "."))
    if (!is.null(x$MSE)){
        writeLines(paste0("MSE: ",round(x$MSE,4),"."))
    }

}

predict.elm.fast.internal <- function(x,w.in,w.out,b,w.dct,direct){
  y <- fast.sig(cbind(1,x) %*% w.in) %*% w.out + b + if(direct!=TRUE){0}else{x %*% w.dct}
  return(y)
}
