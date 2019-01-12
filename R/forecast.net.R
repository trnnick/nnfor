
#' @rdname forecast.elm
#' @title Forecast using ELM neural network.
#' @description Create forecasts using ELM neural networks.
#'
#' @param object ELM network object, produced using \code{\link{elm}}.
#' @param h Forecast horizon. If NULL then h is set to match frequency of time series.
#' @param y Optionally forecast using different data than what the network was trained on. Expected to create havoc and do really bad things!
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set plus the forecast horizon, but can be longer. Set it to NULL if no xreg inputs are used.
#' @param ... Unused argument.
#'
#' @return An object of classes "\code{forecast.net}" and "\code{forecast}".
#'   The function \code{plot} produces a plot of the forecasts.
#'   An object of class \code{"forecast.net"} is a list containing the following elements:
#' \itemize{
#' \item{\code{method}{ - The name of the forecasting method as a character string}}
#' \item{\code{mean}{ - Point forecasts as a time series}}
#' \item{\code{all.mean}{ - An array h x reps of all ensemble members forecasts, where reps are the number of ensemble members.}}
#' \item{\code{x}{ - The original time series used to create the network.}}
#' \item{\code{fitted}{ - Fitted values.}}
#' \item{\code{residuals}{ - Residuals from the fitted network.}}
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}, \code{\link{elm.thief}}, \code{\link{mlp}}.
#' @keywords elm thief ts
#'
#' @examples
#' \dontshow{
#'  fit <- elm(AirPassengers,reps=1)
#'  frc <- forecast(fit,h=36)
#'  print(frc)
#'  plot(frc)
#' }
#' \dontrun{
#'  fit <- elm(AirPassengers)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#'
#' @export
#' @method forecast elm

forecast.elm <- function(object,h=NULL,y=NULL,xreg=NULL,...){
  forecast.net(object,h=h,y=y,xreg=xreg,...)
}

#' @rdname forecast.mlp
#' @title Forecast using MLP neural network.
#' @description  Create forecasts using MLP neural networks.
#'
#' @param object MLP network object, produced using \code{\link{mlp}}.
#' @param h Forecast horizon. If NULL then h is set to match frequency of time series.
#' @param y Optionally forecast using different data than what the network was trained on. Expected to create havoc and do really bad things!
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set plus the forecast horizon, but can be longer. Set it to NULL if no xreg inputs are used.
#' @param ... Unused argument.
#'
#' @inherit forecast.elm return
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{mlp}}, \code{\link{mlp.thief}}, \code{\link{elm}}.
#' @keywords mlp thief ts
#'
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers,reps=1)
#'  frc <- forecast(fit,h=36)
#'  print(frc)
#'  plot(frc)
#' }
#' \dontrun{
#'  fit <- mlp(AirPassengers)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#'
#' @export
#' @method forecast mlp
#'
forecast.mlp <- function(object,h=NULL,y=NULL,xreg=NULL,...){
  forecast.net(object,h=h,y=y,xreg=xreg,...)
}

forecast.net <- function(object,h=NULL,y=NULL,xreg=NULL,...){
  # Produce forecast with NNs

  if (is.null(y)){
    y <- object$y
  }

  # Get frequency
  ff.ls <- get.ff(y)
  ff <- ff.ls$ff
  ff.n <- ff.ls$ff.n
  rm("ff.ls")

  if (is.null(h)){
    h <- max(ff)
  }

  # Get stuff from object list
  cl.object <- class(object)
  is.elm.fast <- any(cl.object == "elm.fast")
  # Get information when neuralnets is used
  if (!is.elm.fast){
    net <- object$net
    reps <- length(net$weights)
  }
  # Get additional ELM definitions
  if (any(class(object) == "elm")){
    direct <- object$direct
    W.in <- object$W.in
    W <- object$W
    b <- object$b
    W.dct <- object$W.dct
    reps <- length(b)
  } else {
    direct <- FALSE
  }
  # Remaining common definitions
  hd <- object$hd
  lags <- object$lags
  xreg.lags <- object$xreg.lags
  difforder <- object$difforder
  sdummy <- object$sdummy
  det.type <- object$det.type
  minmax <- object$minmax
  xreg.minmax <- object$xreg.minmax
  comb <- object$comb
  fitted <- object$fitted
  ff.det <- object$ff.det
  ff.n.det <- length(ff.det)

  # Temporal aggregation can mess-up start/end of ts, so lets fix it
  fend <- end(y)
  if (length(fend)==1){
    fstart <- fend + deltat(y)
  } else {
    fstart <- fend[1] + (fend[2]-1)/frequency(y) + deltat(y)
  }

  # Check xreg inputs
  if (!is.null(xreg)){
    x.n <- dim(xreg)[2]
    if (length(xreg.lags) != x.n){
      stop("Number of xreg inputs is not consistent with network specification (number of xreg.lags).")
    }
    if (dim(xreg)[1] < length(y)+h){
      stop("Length of xreg must be longer that y + forecast horizon.")
    }
  } else {
    x.n <- 0
  }

  # Apply differencing
  d <- length(difforder)
  y.d <- y.ud <- vector("list",d+1)
  y.d[[1]] <- y
  names(y.d)[1] <- "d0"
  if (d>0){
    for (i in 1:d){
      y.d[[i+1]] <- diff(y.d[[i]],difforder[i])
      names(y.d)[i+1] <- paste0(names(y.d)[i],"d",difforder[i])
    }
  }
  Y <- as.vector(linscale(y.d[[d+1]],minmax=minmax)$x)

  # Scale xreg
  if (x.n > 0){
    xreg.sc <- xreg
    for (i in 1:x.n){
      xreg.sc[,i] <- linscale(xreg[,i],minmax=xreg.minmax[[i]])$x
    }
    # Starting point of xreg
    xstart <- length(y)+1
  }

  if (sdummy == TRUE){
    temp <- ts(1:h,start=fstart,frequency=max(ff.det))
    Xd <- vector("list",ff.n.det)
    for (s in 1:ff.n.det){
      if (det.type=="trg"){
        # There was a problem when the fractional seasonalities were < 3, so this is now separated
        Xd[[s]] <- tsutils::seasdummy(h,m=ff.det[s],y=temp,type="trg",full=TRUE)
        Xd[[s]] <- Xd[[s]][,1:min(length(Xd[[s]][1,]),2)]
      } else {
        Xd[[s]] <- tsutils::seasdummy(h,m=ff.det[s],y=temp,type="bin")
      }
      colnames(Xd[[s]]) <- paste0("D",s,".",1:length(Xd[[s]][1,]))
    }
    Xd <- do.call(cbind,Xd)
    # Xd <- seasdummy(h,y=temp,type=det.type)
  }

  Yfrc <- array(NA,c(h,reps),dimnames=list(paste0("t+",1:h),paste0("NN.",1:reps)))
  if (length(lags)>0){
    ylag <- max(lags)
  } else {
    ylag <- 0
  }

  # For each repetition
  for (r in 1:reps){

    frc.sc <- vector("numeric",h)
    for (i in 1:h){

      # Construct inputs
      if (i == 1){
        temp <- NULL
      } else {
        temp <- frc.sc[1:(i-1)]
      }
      xi <- rev(tail(c(Y,temp),ylag)) # Reverse for lags
      xi <- xi[lags]
      # Construct xreg inputs
      if (x.n > 0){
        Xreg <- vector("list",x.n)
        for (j in 1:x.n){
          if (length(xreg.lags[[j]])>0){
            xreg.temp <- xreg.sc[(xstart+i-1):(xstart-max(xreg.lags[[j]])+i-1),j] # Reversing is happening in the indices
            Xreg[[j]] <- xreg.temp[xreg.lags[[j]]+1]
          }
        }
        Xreg.all <- unlist(Xreg)
        xi <- c(xi,Xreg.all)
      }
      xi <- rbind(xi)
      # Construct seasonal dummies inputs
      if (sdummy == TRUE){
        xi <- cbind(xi,Xd[i,,drop=FALSE])
      }

      # Calculate forecasts
      if (any(class(object) == "mlp")){
        yhat.sc <- neuralnet::compute(net,xi,r)$net.result
      } else {
        # EML
        if (is.elm.fast){
          yhat.sc <- predict.elm.fast.internal(xi,W.in[[r]],W[[r]],b[r],W.dct[[r]],direct)
        } else {
          H <- t(as.matrix(tail(neuralnet::compute(net,xi,r)$neurons,1)[[1]][,2:(tail(hd,1)+1)]))
          yhat.sc <- H %*% W[[r]] + b[r] + if(direct!=TRUE){0}else{xi %*% W.dct[[r]]}
        }

      }
      frc.sc[i] <- yhat.sc
    }
    # Reverse scaling
    frc <- linscale(frc.sc,minmax,rev=TRUE)$x

    # Reverse differencing
    f.ud <- vector("list",d+1)
    names(f.ud) <- names(y.ud)
    f.ud[[d+1]] <- frc
    if (d>0){
      for (i in 1:d){
        temp <- c(tail(y.d[[d+1-i]],difforder[d+1-i]),f.ud[[d+2-i]])
        n.t <- length(temp)
        for (j in 1:(n.t-difforder[d+1-i])){
          temp[difforder[d+1-i]+j] <- temp[j] + temp[difforder[d+1-i]+j]
        }
        f.ud[[d+1-i]] <- temp[(difforder[d+1-i]+1):n.t]
      }
    }
    fout <- head(f.ud,1)[[1]]

    Yfrc[,r] <- fout
  }

  # Combine forecasts
  fout <- frc.comb(Yfrc,comb)

  fout <- ts(fout,frequency=frequency(y),start=fstart)

  # Prepare output
  out <- list("method"=class(object),"mean"=fout,
              "all.mean"=ts(Yfrc,frequency=frequency(y),start=fstart),
              "x"=y,"fitted"=fitted,"residuals"=y-fitted)
  return(structure(out,class=c("forecast.net","forecast")))

}
