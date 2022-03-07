#' Extreme learning machines for time series forecasting
#'
#' This function fits ELM neural networks for time series forecasting.
#'
#' @param y Input time series. Can be ts or msts object.
#' @param m Frequency of the time series. By default it is picked up from y.
#' @param hd Number of hidden nodes. This can be a vector, where each number represents the number of hidden nodes of a different hidden layer. Use NULL to automatically specify.
#' @param type Estimation type for output layer weights. Can be "lasso" (lasso with CV), "ridge" (ridge regression with CV), "step" (stepwise regression with AIC) or "lm" (linear regression).
#' @param reps Number of networks to train, the result is the ensemble forecast.
#' @param comb Combination operator for forecasts when reps > 1. Can be "median", "mode" (based on KDE estimation) and "mean".
#' @param lags Lags of y to use as inputs. If none provided then 1:frequency(y) is used. Use 0 for no univariate lags.
#' @param keep Logical vector to force lags to stay in the model if sel.lag == TRUE. If NULL then it keep = rep(FALSE,length(lags)).
#' @param difforder Vector including the differencing lags. For example c(1,12) will apply first and seasonal (12) differences. For no differencing use 0. For automatic selection use NULL.
#' @param outplot Provide plot of model fit. Can be TRUE or FALSE.
#' @param sel.lag Automatically select lags. Can be TRUE or FALSE.
#' @param direct Use direct input-output connections to model strictly linear effects. Can be TRUE or FALSE.
#' @param allow.det.season Permit modelling seasonality with deterministic dummies.
#' @param det.type Type of deterministic seasonality dummies to use. This can be "bin" for binary or "trg" for a sine-cosine pair. With "auto" if ony a single seasonality is used and periodicity is up to 12 then "bin" is used, otherwise "trg".
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set, but can be longer.
#' @param xreg.lags This is a list containing the lags for each exogenous variable. Each list is a numeric vector containing lags. If xreg has 3 columns then the xreg.lags list must contain three elements. If NULL then it is automatically specified.
#' @param xreg.keep List of logical vectors to force lags of xreg to stay in the model if sel.lag == TRUE. If NULL then all exogenous lags can be removed.
#' @param barebone Use an alternative elm implementation (written in R) that is faster when the number of inputs is very high. Typically not needed.
#' @param model A previously trained mlp object. If this is provided then the same model is fitted to y, without re-estimating any model parameters.
#' @param retrain If a previous model is provided, retrain the network or not. If the network is retrained the size of the hidden layer is reset.
#'
#' @return Return object of class \code{elm}. If barebone == TRUE then the object inherits a second class "\code{elm.fast}".
#'   The function \code{plot} produces a plot the network architecture.
#'   \code{elm} contains:
#' \itemize{
#' \item{\code{net}{ - ELM networks. If it is of class "\code{elm.fast}" then this is NULL.}}
#' \item{\code{hd}{ - Number of hidden nodes. If it is of class "\code{elm.fast}" this is a vector with a different number for each repetition.}}
#' \item{\code{W.in}{ - NULL unless it is of class "\code{elm.fast}". Contains the input weights.}}
#' \item{\code{W}{ - Output layer weights for each repetition.}}
#' \item{\code{b}{ - Contains the output node bias for each training repetition.}}
#' \item{\code{W.dct}{ - Contains the direct connection weights if argument direct == TRUE. Otherwise is NULL.}}
#' \item{\code{lags}{ - Input lags used.}}
#' \item{\code{xreg.lags}{ - \code{xreg} lags used.}}
#' \item{\code{difforder}{ - Differencing used.}}
#' \item{\code{sdummy}{ - Use of deterministic seasonality.}}
#' \item{\code{ff}{ - Seasonal frequencies detected in data (taken from ts or msts object).}}
#' \item{\code{ff.det}{ - Seasonal frequencies coded using deterministic dummies.}}
#' \item{\code{det.type}{ - Type of determistic seasonality.}}
#' \item{\code{y}{ - Input time series.}}
#' \item{\code{minmax}{ - Scaling structure.}}
#' \item{\code{xreg.minmax}{ - Scaling structure for xreg variables.}}
#' \item{\code{comb}{ - Combination operator used.}}
#' \item{\code{type}{ - Estimation used for output layer weights.}}
#' \item{\code{direct}{ - Presence of direct input-output connections.}}
#' \item{\code{fitted}{ - Fitted values.}}
#' \item{\code{MSE}{ - In-sample Mean Squared Error.}}
#' }
#'
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{forecast.elm}}, \code{\link{elm.thief}}, \code{\link{mlp}}.
#' @references
#' \itemize{
#' \item{For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{http://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.}
#' \item{For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{http://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.}
#' \item{For variable selection see: Crone S.F., Kourentzes N. (2010) \href{http://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction – A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.}
#' \item{For ELMs see: Huang G.B., Zhou H., Ding X. (2006) Extreme learning machine: theory and applications. \emph{Neurocomputing}, \bold{70}(\bold{1}), 489-501.}
#' }
#' @keywords mlp thief ts
#' @note To use elm with Temporal Hierarchies (\href{https://cran.r-project.org/package=thief}{thief} package) see \code{\link{elm.thief}}.
#'   The elm function by default calls the \code{neuralnet} function. If barebone == TRUE then it uses an alternative implementation (\code{TStools:::elm.fast}) which is more appropriate when the number of inputs is several hundreds.
#' @examples
#' \dontshow{
#'  fit <- elm(AirPassengers,reps=1)
#'  print(fit)
#'  plot(fit)
#' }
#' \dontrun{
#'  fit <- elm(AirPassengers)
#'  print(fit)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#'
#' @export
elm <- function(y,m=frequency(y),hd=NULL,type=c("lasso","ridge","step","lm"),reps=20,comb=c("median","mean","mode"),
                lags=NULL,keep=NULL,difforder=NULL,outplot=c(FALSE,TRUE),sel.lag=c(TRUE,FALSE),direct=c(FALSE,TRUE),
                allow.det.season=c(TRUE,FALSE),det.type=c("auto","bin","trg"),
                xreg=NULL,xreg.lags=NULL,xreg.keep=NULL,barebone=c(FALSE,TRUE),model=NULL,retrain=c(FALSE,TRUE)){

  # Defaults
  type <- match.arg(type,c("lasso","ridge","step","lm"))
  comb <- match.arg(comb,c("median","mean","mode"))
  outplot <- outplot[1]
  sel.lag <- sel.lag[1]
  direct <- direct[1]
  allow.det.season <- allow.det.season[1]
  det.type <- det.type[1]
  barebone <- barebone[1]
  retrain <- retrain[1]

  # Check if y input is a time series
  if (!(any(class(y) == "ts") | any(class(y) == "msts"))){
    stop("Input y must be of class ts or msts.")
  }

  # Check if a model input is provided
  if (!is.null(model)){
    if (any(class(model)=="elm")){
      oldmodel <- TRUE
      if (retrain == FALSE){
        hd <- model$hd
      }
      lags <- model$lags
      xreg.lags <- model$xreg.lags
      difforder <- model$difforder
      comb <- model$comb
      det.type <- model$det.type
      allow.det.season <- model$sdummy
      direct <- model$direct
      type <- model$type
      reps <- length(model$b)
      sel.lag <- FALSE
      if (any(class(model)=="elm.fast")){
        barebone <- TRUE
      }
      # Check xreg inputs
      x.n <- dim(xreg)[2]
      if (is.null(x.n)){
        x.n <- 0
      }
      xm.n <- length(model$xreg.minmax)
      if (x.n != xm.n){
        stop("Previous model xreg specification and new xreg inputs do not match.")
      }
    } else {
      stop("model must be an mlp object, the output of the mlp() function.")
    }
  } else {
    oldmodel <- FALSE
  }

  # Check xreg inputs
  if (!is.null(xreg)){
    x.n <- length(xreg[1,])
    if (!is.null(xreg.lags)){
      if (length(xreg.lags) != x.n){
        stop("Argument xreg.lags must be a list with as many elements as xreg variables (columns).")
      }
    }
  }

  # Get frequency
  ff.ls <- get.ff(y)
  ff <- ff.ls$ff
  ff.n <- ff.ls$ff.n
  rm("ff.ls")
  # Check seasonality of old model
  if (oldmodel == TRUE){
    if (model$sdummy == TRUE){
      if (!all(model$ff.det == ff)){
        stop("Seasonality of current data and seasonality of provided model does not match.")
      }
    }
  }

  # Default lagvector
  xreg.ls <- def.lags(lags,keep,ff,xreg.lags,xreg.keep,xreg)
  lags <- xreg.ls$lags
  keep <- xreg.ls$keep
  xreg.lags <- xreg.ls$xreg.lags
  xreg.keep <- xreg.ls$xreg.keep
  rm("xreg.ls")

  # Pre-process data (same for MLP and ELM)
  PP <- preprocess(y,m,lags,keep,difforder,sel.lag,allow.det.season,det.type,ff,ff.n,xreg,xreg.lags,xreg.keep)
  Y <- PP$Y
  X <- PP$X
  sdummy <- PP$sdummy
  difforder <- PP$difforder
  det.type <- PP$det.type
  lags <- PP$lags
  xreg.lags <- PP$xreg.lags
  sc <- PP$sc
  xreg.minmax <- PP$xreg.minmax
  d <- PP$d
  y.d <- PP$y.d
  y.ud <- PP$y.ud
  frm <- PP$frm
  ff.det <- PP$ff.det
  lag.max <- PP$lag.max
  rm("PP")

  if (is.null(hd)){
    hd <- min(100-60*(type=="step" | type=="lm"),max(4,length(Y)-2-as.numeric(direct)*length(X[1,])))
  }

  # Train ELM
  # If single hidden layer switch to fast, unless requested otherwise
  if ((length(hd)==1 & barebone == TRUE) | (barebone == TRUE & oldmodel == TRUE)){
    if (oldmodel == FALSE | retrain == TRUE){

      # Train model
      # Switch to elm.fast
      f.elm <- elm.fast(Y,X,hd=hd,reps=reps,comb=comb,type=type,direct=direct,linscale=FALSE,output="linear",core=TRUE)
      # Post-process output
      Yhat <- f.elm$fitted.all

    } else {

      # Use inputted model
      Yhat <- array(NA,c(length(Y),reps))
      W.in <- model$W.in
      W <- model$W
      B <- model$b
      W.dct <- model$W.dct
      for (r in 1:reps){
        Yhat[,r] <- predict.elm.fast.internal(X,W.in[[r]],W[[r]],B[r],W.dct[[r]],direct)
      }

    }
    for (r in 1:reps){
      # Reverse scaling
      yhat <- linscale(Yhat[,r],sc$minmax,rev=TRUE)$x
      # Undifference - this is 1-step ahead undifferencing
      y.ud[[d+1]] <- yhat
      if (d>0){
        for (i in 1:d){
          n.ud <- length(y.ud[[d+2-i]])
          n.d <- length(y.d[[d+1-i]])
          y.ud[[d+1-i]] <- y.d[[d+1-i]][(n.d-n.ud-difforder[d+1-i]+1):(n.d-difforder[d+1-i])] + y.ud[[d+2-i]]
        }
      }
      Yhat[,r] <- head(y.ud,1)[[1]]
    }

  } else {
    # Rely on neuralnet, very slow when number of inputs is large
    if (oldmodel == FALSE | retrain == TRUE){
      # Train network
      net <- neuralnet::neuralnet(frm,cbind(Y,X),hidden=hd,threshold=10^10,rep=reps,err.fct="sse",linear.output=FALSE)
    } else {
      net <- model$net
    }

      # Get weights for each repetition
      W <- W.dct <- vector("list",reps)
      B <- vector("numeric",reps)
      Yhat <- array(NA,c((length(y)-sum(difforder)-lag.max),reps))
      x.names <- colnames(X)

      for (r in 1:reps){

        H <- as.matrix(tail(neuralnet::compute(net,X,r)$neurons,1)[[1]][,2:(tail(hd,1)+1)])
        if (direct==TRUE){
          Z <- cbind(H,X)
        } else {
          Z <- H
        }

        # Train network (last layer)
        if (oldmodel == FALSE | retrain == TRUE){
          w.out <- elm.train(Y,Z,type,X,direct,hd,output="linear")
          B[r] <- w.out[1]                                  # Bias (Constant)
          if (direct == TRUE){                              # Direct connections
            w.dct <- w.out[(1+hd+1):(1+hd+dim(X)[2]),,drop=FALSE]
            if (!is.null(x.names)){
              rownames(w.dct) <- x.names
            }
            W.dct[[r]] <- w.dct
          }
          W[[r]] <- w.out[2:(1+hd),,drop=FALSE]             # Hidden layer
        } else {
          # Take last layer weights from imputted model
          B[r] <- model$b[r]
          W.dct[[r]] <- model$W.dct[[r]]
          W[[r]] <- model$W[[r]]
        }

        # Produce fit
        yhat.sc <- H %*% W[[r]] + B[r] + if(direct!=TRUE){0}else{X %*% W.dct[[r]]}

        # Post-process
        yhat <- linscale(yhat.sc,sc$minmax,rev=TRUE)$x

        # # Check unscaled, but differenced fit
        # plot(1:length(tail(y.d,1)[[1]]),tail(y.d,1)[[1]], type="l")
        # lines((max(lags)+1):length(tail(y.d,1)[[1]]),yhat,col="red")

        # Undifference - this is 1-step ahead undifferencing
        y.ud[[d+1]] <- yhat
        if (d>0){
          for (i in 1:d){
            n.ud <- length(y.ud[[d+2-i]])
            n.d <- length(y.d[[d+1-i]])
            y.ud[[d+1-i]] <- y.d[[d+1-i]][(n.d-n.ud-difforder[d+1-i]+1):(n.d-difforder[d+1-i])] + y.ud[[d+2-i]]
          }
        }
        yout <- head(y.ud,1)[[1]]

        # yout <- ts(yout,end=end(y),frequency=frequency(y))
        # # Check undifferences and unscaled
        # plot(y)
        # lines(yout,col="red")
        # # plot(1:length(y),y,type="l")
        # # lines((max(lags)+1+sum(difforder)):length(y),y.ud[[1]],col="red")

        Yhat[,r] <- yout

    } # Close reps

  } # Close elm.fast if

  # Combine forecasts
  yout <- frc.comb(Yhat,comb)

  # Convert to time series
  yout <- ts(yout,end=end(y),frequency=frequency(y))

  MSE <- mean((y[(lag.max+1+sum(difforder)):length(y)] - yout)^2)

  # Construct plot
  if (outplot==TRUE){

    plot(y)
    if (reps>1){
      for (i in 1:reps){
        temp <- Yhat[,i]
        temp <- ts(temp,frequency=frequency(y),end=end(y))
        lines(temp,col="grey")
      }
    }
    lines(yout,col="blue")

  }

  if (barebone == FALSE){
    W.in <- NULL
    class.type <- "elm"
  } else {
    net <- NULL
    if (oldmodel == FALSE || retrain == TRUE){
      hd <- f.elm$hd
      W <- f.elm$W
      B <- f.elm$b
      W.in <- f.elm$W.in
      W.dct <- f.elm$W.dct
    }
    class.type <- c("elm","elm.fast")
  }

  return(structure(list("net"=net,"hd"=hd,"W.in"=W.in,"W"=W,"b"=B,"W.dct"=W.dct,
                        "lags"=lags,"xreg.lags"=xreg.lags,"difforder"=difforder,
                        "sdummy"=sdummy,"ff.det"=ff.det,"det.type"=det.type,"y"=y,"minmax"=sc$minmax,"xreg.minmax"=xreg.minmax,
                        "comb"=comb,"type"=type,"direct"=direct,"fitted"=yout,"MSE"=MSE),class=class.type))

}

elm.train <- function(Y,Z,type,X,direct,hd,output="linear"){
  # Find output weights for ELM

  switch(type,
         "lasso" = {
           if (output=="logistic"){
             fit <- suppressWarnings(glmnet::cv.glmnet(Z,cbind(Y),family="binomial"))
           } else {
             fit <- suppressWarnings(glmnet::cv.glmnet(Z,cbind(Y)))
           }
           cf <- as.vector(coef(fit))
         },
         "ridge" = {
           if (output=="logistic"){
             fit <- suppressWarnings(glmnet::cv.glmnet(Z,cbind(Y),alpha=0,family="binomial"))
           } else {
             fit <- suppressWarnings(glmnet::cv.glmnet(Z,cbind(Y),alpha=0))
           }
           cf <- as.vector(coef(fit))
         },
         { # LS and stepwise
           reg.data <- as.data.frame(cbind(Y,Z))
           colnames(reg.data) <- c("Y",paste0("X",1:(tail(hd,1)+as.numeric(direct)*length(X[1,]))))
           # Take care of linear dependency
           alias.fit <- alias(as.formula(paste0("Y~",paste0("X",1:(tail(hd,1)+as.numeric(direct)*length(X[1,])),collapse="+"))),data=reg.data)
           alias.x <- rownames(alias.fit$Complete)
           frm <- as.formula(paste0("Y~",paste0(setdiff(colnames(reg.data)[2:(hd+1+as.numeric(direct)*length(X[1,]))],alias.x),collapse="+")))
           fit <- suppressWarnings(lm(frm,reg.data))
           if (type == "step"){
             fit <- suppressWarnings(MASS::stepAIC(fit,trace=0)) # ,direction="backward",k=log(length(Y)))) # BIC criterion
           }
           cf.temp <- coef(fit)
           loc <- which(colnames(reg.data) %in% names(cf.temp))
           cf <- rep(0,(tail(hd,1)+1+direct*length(X[1,])))
           cf[1] <- cf.temp[1]
           cf[loc] <- cf.temp[2:length(cf.temp)]
         })

  return(cbind(cf))

}
