#' Multilayer Perceptron for time series forecasting
#'
#' This function fits MLP neural networks for time series forecasting.
#'
#' @param y Input time series. Can be ts or msts object.
#' @param m Frequency of the time series. By default it is picked up from y.
#' @param hd Number of hidden nodes. This can be a vector, where each number represents the number of hidden nodes of a different hidden layer.
#' @param reps Number of networks to train, the result is the ensemble forecast.
#' @param comb Combination operator for forecasts when reps > 1. Can be "median", "mode" (based on KDE estimation) and "mean".
#' @param lags Lags of y to use as inputs. If none provided then 1:frequency(y) is used. Use 0 for no univariate lags.
#' @param keep Logical vector to force lags to stay in the model if sel.lag == TRUE. If NULL then it keep = rep(FALSE,length(lags)).
#' @param difforder Vector including the differencing lags. For example c(1,12) will apply first and seasonal (12) differences. For no differencing use 0. For automatic selection use NULL.
#' @param outplot Provide plot of model fit. Can be TRUE or FALSE.
#' @param sel.lag Automatically select lags. Can be TRUE or FALSE.
#' @param allow.det.season Permit modelling seasonality with deterministic dummies.
#' @param det.type Type of deterministic seasonality dummies to use. This can be "bin" for binary or "trg" for a sine-cosine pair. With "auto" if ony a single seasonality is used and periodicity is up to 12 then "bin" is used, otherwise "trg".
#' @param xreg Exogenous regressors. Each column is a different regressor and the sample size must be at least as long as the target in-sample set, but can be longer.
#' @param xreg.lags This is a list containing the lags for each exogenous variable. Each list is a numeric vector containing lags. If xreg has 3 columns then the xreg.lags list must contain three elements. If NULL then it is automatically specified.
#' @param xreg.keep List of logical vectors to force lags of xreg to stay in the model if sel.lag == TRUE. If NULL then all exogenous lags can be removed. The syntax for multiple xreg is the same as for xreg.lags.
#' @param hd.auto.type Used only if hd==NULL. "set" fixes hd=5. "valid" uses a 20\% validation set (randomly) sampled to find the best number of hidden nodes. "cv" uses 5-fold cross-validation. "elm" uses ELM to estimate the number of hidden nodes (experimental).
#' @param hd.max When hd.auto.type is set to either "valid" or "cv" then this argument can be used to set the maximum number of hidden nodes to evaluate, otherwise the maximum is set automatically.
#' @param model A previously trained mlp object. If this is provided then the same model is fitted to y, without re-estimating any model parameters.
#' @param retrain If a previous model is provided, retrain the network or not.
#' @param ... Additional inputs for neuralnet function.
#'
#' @return Return object of class \code{mlp}.
#'   The function \code{plot} produces a plot the network architecture.
#'   \code{mlp} contains:
#' \itemize{
#' \item{\code{net}{ - MLP networks.}}
#' \item{\code{hd}{ - Number of hidden nodes.}}
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
#' \item{\code{fitted}{ - Fitted values.}}
#' \item{\code{MSE}{ - In-sample Mean Squared Error.}}
#' \item{\code{MSEH}{ - If \code{hd.auto.type} is set to either "valid" or "cv" an array of the MSE error for each network size is provided. Otherwise this is NULL.}}
#' }
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{forecast.mlp}}, \code{\link{mlp.thief}}, \code{\link{elm}}.
#' @references
#' \itemize{
#' \item{For an introduction to neural networks see: Ord K., Fildes R., Kourentzes N. (2017) \href{https://kourentzes.com/forecasting/2017/10/16/new-forecasting-book-principles-of-business-forecasting-2e/}{Principles of Business Forecasting 2e}. \emph{Wessex Press Publishing Co.}, Chapter 10.}
#' \item{For combination operators see: Kourentzes N., Barrow B.K., Crone S.F. (2014) \href{https://kourentzes.com/forecasting/2014/04/19/neural-network-ensemble-operators-for-time-series-forecasting/}{Neural network ensemble operators for time series forecasting}. \emph{Expert Systems with Applications}, \bold{41}(\bold{9}), 4235-4244.}
#' \item{For variable selection see: Crone S.F., Kourentzes N. (2010) \href{https://kourentzes.com/forecasting/2010/04/19/feature-selection-for-time-series-prediction-a-combined-filter-and-wrapper-approach-for-neural-networks/}{Feature selection for time series prediction – A combined filter and wrapper approach for neural networks}. \emph{Neurocomputing}, \bold{73}(\bold{10}), 1923-1936.}
#' }
#' @keywords mlp thief ts
#' @note To use mlp with Temporal Hierarchies (\href{https://cran.r-project.org/package=thief}{thief} package) see \code{\link{mlp.thief}}.
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers,reps=1)
#'  print(fit)
#'  plot(fit)
#' }
#' \dontrun{
#'  fit <- mlp(AirPassengers)
#'  print(fit)
#'  plot(fit)
#'  frc <- forecast(fit,h=36)
#'  plot(frc)
#' }
#'
#' @export mlp

mlp <- function(y,m=frequency(y),hd=NULL,reps=20,comb=c("median","mean","mode"),
                lags=NULL,keep=NULL,difforder=NULL,outplot=c(FALSE,TRUE),sel.lag=c(TRUE,FALSE),
                allow.det.season=c(TRUE,FALSE),det.type=c("auto","bin","trg"),
                xreg=NULL, xreg.lags=NULL,xreg.keep=NULL,hd.auto.type=c("set","valid","cv","elm"),
                hd.max=NULL, model=NULL, retrain=c(FALSE,TRUE), ...){

    # hd.max is only relevant to valid and cv

    # Defaults
    comb <- match.arg(comb,c("median","mean","mode"))
    outplot <- outplot[1]
    sel.lag <- sel.lag[1]
    allow.det.season <- allow.det.season[1]
    det.type <- det.type[1]
    hd.auto.type <- hd.auto.type[1]
    retrain <- retrain[1]

    # Check if y input is a time series
    if (!(is(y,"ts") | is(y,"msts"))){
      stop("Input y must be of class ts or msts.")
    }

    # Check if a model input is provided
    if (!is.null(model)){
      if (is(model,"mlp")){
        oldmodel <- TRUE
        hd <- model$hd
        lags <- model$lags
        xreg.lags <- model$xreg.lags
        difforder <- model$difforder
        comb <- model$comb
        det.type <- model$det.type
        allow.det.season <- model$sdummy
        reps <- length(model$net$weights)
        sel.lag <- FALSE
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
        stop("model must be a mlp object, the output of the mlp() function.")
      }
    } else {
      oldmodel <- FALSE
    }

    # Check xreg inputs
    if (!is.null(xreg)){
      x.n <- dim(xreg)[2]
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

    # Auto specify number of hidden nodes
    mseH <- NULL
    if (is.null(hd)){
      switch(hd.auto.type,
             "set" = {hd <- 5},
             "valid" = {hd <- auto.hd.cv(Y,X,frm,comb,reps,type="valid",hd.max)
                        mseH <- hd$mseH; hd <- hd$hd},
             "cv" = {hd <- auto.hd.cv(Y,X,frm,comb,reps,type="cv",hd.max)
                     mseH <- hd$mseH; hd <- hd$hd},
             "elm" = {hd <- auto.hd.elm(Y,X,frm)}
             )
    }

    # Create network
    if (oldmodel == FALSE | retrain == TRUE){
      net <- neuralnet::neuralnet(frm,cbind(Y,X),hidden=hd,rep=reps,err.fct="sse",linear.output=TRUE,...)
    } else {
      net <- model$net
    }
    # In case some networks did not train reduce the number of available repetitions
    reps <- length(net$weights)

    # Produce forecasts
    Yhat <- array(NA,c((length(y)-sum(difforder)-lag.max),reps))

    for (r in 1:reps){

        yhat.sc <- neuralnet::compute(net,X,r)$net.result
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

    resid <- (y - yout)

    return(structure(list("net"=net,"hd"=hd,"lags"=lags,"xreg.lags"=xreg.lags,"difforder"=difforder,"sdummy"=sdummy,"ff.det"=ff.det,
                          "det.type"=det.type,"y"=y,"minmax"=sc$minmax,"xreg.minmax"=xreg.minmax,"comb"=comb,"fitted"=yout,
                          "residuals" = resid,"MSE"=MSE,"MSEH"=mseH),class="mlp"))

}

get.ff <- function(y){
  # Get time series frequency
  if (is(y,"msts")){
    ff <- attributes(y)$msts
    ff.n <- length(ff)
  } else {
    ff <- frequency(y)
    ff.n <- 1
  }
  return(list("ff"=ff,"ff.n"=ff.n))
}

def.lags <- function(lags,keep,ff,xreg.lags,xreg.keep,xreg){
  # Default lagvector
  if (is.null(lags)){
    if (max(ff)>3){
      lags <- 1:max(ff)
    } else {
      lags <- 1:4
    }
  }
  if (!is.null(xreg) && is.null(xreg.lags)){
    x.n <- dim(xreg)[2]
    xreg.lags <- rep(list(lags),x.n)
  }

  # Check keep options
  if (!is.null(keep)){
    if (length(keep) != length(lags)){
      stop("Argument `keep' must be a logical vector of length equal to lags.")
    }
  } else {
    keep <- rep(FALSE,length(lags))
  }
  # If no univariate lags are requested, then make keep=NULL
  if (length(lags)==1 & lags[1]==0){
    keep <- NULL
  }
  if (!is.null(xreg.keep)){
    if (all(unlist(lapply(xreg.lags,length)) == unlist(lapply(xreg.keep,length)))==FALSE){
      stop("Argument `xreg.keep' must be a list of logical vectors of length equal to the length of the lags in xreg.lags.")
    }
  } else {
    xreg.keep <- lapply(unlist(lapply(xreg.lags,length)),function(x){rep(FALSE,x)})
  }

  return(list("lags"=lags,"keep"=keep,"xreg.lags"=xreg.lags,"xreg.keep"=xreg.keep))
}

preprocess <- function(y,m,lags,keep,difforder,sel.lag,allow.det.season,det.type,ff,ff.n,xreg,xreg.lags,xreg.keep){
# Pre-process data for MLP and ELM

  # Check seasonality & trend
  cma <- tsutils::cmav(y,ma=max(ff))
  st <- seasoncheck(y,m=max(ff),cma=cma)
  if (is.null(st$season.exist)){
    st$season.exist <- FALSE
  }

  # Specify differencing order
  difforder <- ndiffs.net(difforder,y,ff,st)

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
  names(y.ud) <- names(y.d)

  # Scale target
  sc <- linscale(tail(y.d,1)[[1]],minmax=list("mn"=-.8,"mx"=0.8))
  y.sc <- sc$x
  n <- length(y.sc)

  # Scale xregs and trim initial values for differencing of y
  if (!is.null(xreg)){
    x.n <- dim(xreg)[2]
    xreg.sc <- array(NA,c(dim(xreg)[1]-sum(difforder),x.n))
    xreg.minmax <- vector("list",x.n)
    dstart <- sum(difforder)
    xreg <- xreg[(sum(difforder)+1):dim(xreg)[1],,drop=FALSE]
    for (i in 1:x.n){
      xreg.sc.temp <- linscale(xreg[,i],minmax=list("mn"=-.8,"mx"=0.8))
      xreg.sc[,i] <- xreg.sc.temp$x
      xreg.minmax[[i]] <- xreg.sc.temp$minmax
    }
  } else {
    x.n <- 0
    xreg.minmax <- NULL
    xreg.sc <- xreg
  }

  net.inputs <- create.inputs(y.sc, xreg.sc, lags, xreg.lags, n)
  Y <- net.inputs$Y
  X <- net.inputs$X
  Xreg <- net.inputs$Xreg
  lag.max <- net.inputs$lag.max
  rm("net.inputs")

  # Create seasonal dummies
  seas.dum <- seas.dum.net(st,difforder,det.type,ff,ff.n,Y,y,allow.det.season)
  Xd <- seas.dum$Xd
  det.type <- seas.dum$det.type
  sdummy <- seas.dum$sdummy
  rm("seas.dum")

  # Select lags
  if (sel.lag == TRUE){
    if (x.n>0){
      Xreg.all <- do.call(cbind,Xreg)
      Xreg.all <- Xreg.all[1:length(Y),,drop=FALSE]
    } else {
      Xreg.all <- NULL
    }
    reg.isel <- as.data.frame(cbind(Y,X,Xreg.all))
    # colnames(reg.isel) <- c("Y",paste0("X",lags),paste0("Xreg",))
    if (sdummy == FALSE){
      # Check if there are no inputs at all
      if (all(colnames(reg.isel) == "Y")){
        stop("Cannot build a network with no univariate or exogenous lags and no deterministic seasonality. Increase the maximum lags.")
      } else {
        fit <- lm(formula=Y~.,data=reg.isel)
        if (sdummy == FALSE){
          ff.det <- NULL
        }
      }
    } else {
      lm.frm <- as.formula(paste0("Y~.+",paste(paste0("Xd[[",1:ff.n,"]]"),collapse="+")))
      fit <- lm(formula=lm.frm,data=reg.isel)
    }
    # Make selection of lags robust to sample size issues if all
    # other checks fail by using lasso
    cf.temp <- tryCatch({
      keep.all <- c(keep,unlist(xreg.keep),rep(FALSE,ff.n))
      if (all(!keep.all)){
        scp <- NULL
      } else {
        scp <- list(lower = as.formula(paste("~",paste(attributes(fit$terms)$term.labels[keep.all],collapse=" + "))))
      }
      fit <- MASS::stepAIC(fit,trace=0,direction="backward",scope=scp)
      # Get useful lags
      cf.temp <- coef(fit)
    }, error = function(e) {
      lasso.Y <- reg.isel[,1]
      lasso.X <- data.matrix(reg.isel[,2:dim(reg.isel)[2],drop=FALSE])
      if (sdummy == TRUE){
        for (i in 1:ff.n){
          tempX <- Xd[[i]]
          colnames(tempX) <- paste0(paste0("Xd[[",i,"]]"),colnames(tempX))
          lasso.X <- cbind(lasso.X,tempX)
        }
      }
      if (any(keep.all)){
          warning("Cannot execute backwards variable selection, reverting to Lasso. Arguments `keep' and `xreg.keep' will be ignored.")
      }
      fit.lasso <- suppressWarnings(glmnet::cv.glmnet(x=lasso.X,y=lasso.Y))
      cf.temp <- as.vector(coef(fit.lasso))
      names(cf.temp) <- rownames(coef(fit.lasso))
      cf.temp <- cf.temp[cf.temp!=0]
    })
    X.loc <- lags[which(colnames(X) %in% names(cf.temp))]

    if (x.n>0){
      Xreg.loc <- xreg.lags
      for (i in 1:x.n){
        Xreg.loc[[i]] <- xreg.lags[[i]][which(colnames(Xreg[[i]]) %in% names(cf.temp))]
      }
      xreg.lags <- Xreg.loc
    }

    # Check if deterministic seasonality has remained in the model
    if (sdummy == TRUE){
      still.det <- rep(TRUE,ff.n)
      # Trigonometric dummies will not be retained by linear regression
      # so do not allow rejection by stepwise!
      if (det.type == "bin"){
        for (i in 1:ff.n){
          still.det[i] <- any(grepl(paste0("Xd[[",i,"]]"),names(cf.temp),fixed=TRUE))
        }
      }
    }

    # Although there is an error above to avoid having no inputs, it
    # may still happen if regession rejects all lags. Give a warning!
    # Check if there are any lags
    if (x.n>0){
      # If not univariate and exogenous
      if (sum(c(length(X.loc),unlist(lapply(Xreg.loc,length))))==0){
        # If no deterministic seasonal
        if (sdummy == FALSE){
          warning("No inputs left in the network after pre-selection, forcing AR(1).")
          X.loc <- 1
        }
      }
    } else {
      # If no univariate lags
      if (length(X.loc)==0){
        # If no deterministic seasonal
        if (sdummy == FALSE){
          warning("No inputs left in the network after pre-selection, forcing AR(1).")
          X.loc <- 1
        }
      }
    }
    if (length(X.loc)>0){
      lags <- X.loc
    }

    # Recreate inputs
    net.inputs <- create.inputs(y.sc, xreg.sc, lags, xreg.lags, n)
    Y <- net.inputs$Y
    X <- net.inputs$X
    Xreg <- net.inputs$Xreg
    lag.max <- net.inputs$lag.max
    rm("net.inputs")

    # Recreate seasonal dummies
    if (sdummy == TRUE){
        # Re-create seasonal dummies
        if (sum(still.det)==0){
            sdummy <- FALSE
            ff.det <- NULL
        } else {
            ff.det <- ff[still.det]
            ff.n.det <- length(ff.det)
            Xd <- vector("list",ff.n.det)
            for (s in 1:ff.n.det){
              if (det.type=="trg"){
                # There was a problem when the fractional seasonalities were < 3, so this is now separated
                Xd[[s]] <- tsutils::seasdummy(length(Y),y=ts(Y,end=end(y),frequency=ff[s]),type="trg",full=TRUE)
                Xd[[s]] <- Xd[[s]][,1:min(length(Xd[[s]][1,]),2)]
              } else {
                Xd[[s]] <- tsutils::seasdummy(length(Y),y=ts(Y,end=end(y),frequency=ff[s]),type="bin")
              }
              colnames(Xd[[s]]) <- paste0("D",s,".",1:length(Xd[[s]][1,]))
            }
        }
    }

  } else {
    # If no selection is done, match frequencies of dummies with frequencies of time series
    ff.det <- ff
  }

  # Merge lags and deterministic seasonality to create network inputs
  if (x.n>0){
    Xreg.all <- do.call(cbind,Xreg)
  } else {
    Xreg.all <- NULL
  }
  X.all <- cbind(X,Xreg.all[1:length(Y),,drop=FALSE])
  if (sdummy == TRUE){
    Xd <- do.call(cbind,Xd)
    X.all <- cbind(X.all,Xd)
  }

  # Network formula
  frm <- paste0(colnames(X.all),collapse="+")
  frm <- as.formula(paste0("Y~",frm))

  return(list("Y"=Y,"X"=X.all,"sdummy"=sdummy,"difforder"=difforder,"det.type"=det.type,"lags"=lags,"xreg.lags"=xreg.lags,"lag.max"=lag.max,"sc"=sc,"xreg.minmax"=xreg.minmax,"d"=d,"y.d"=y.d,"y.ud"=y.ud,"frm"=frm,"ff.det"=ff.det))

}

ndiffs.net <- function(difforder,y,ff,st){
  # Find differencing for neural nets
  # NULL is automatic
  # 0 is no differencing

  # Find differencing order
  if (all(difforder != 0)){
    cma <- st$cma
    if (is.null(difforder)){
      # Identify difforder automatically
      difforder <- 0
      if (trendcheck(cma) == TRUE){
        difforder <- 1
      }
      if (frequency(y)>1){
        if (st$season.exist == TRUE){
          # difforder <- c(difforder,frequency(y))

          # Remove trend appropriately
          if (length(y)/max(ff) < 3){
            m.seas <- TRUE
          } else {
            # Test can only run if there are at least three seasons
            m.seas <- tsutils::mseastest(y,m=max(ff),cma=cma)$is.multiplicative
          }
          if (m.seas == TRUE){
            y.dt <- y/cma
          } else {
            y.dt <- y-cma
          }
          d.order <- forecast::nsdiffs(ts(y.dt,frequency=max(ff)),test="ch")
          if (d.order > 0){
            difforder <- c(difforder,max(ff))
          }
        }
      }
    }
  }

    # To remove differencing from the remaining it should be set to NULL
    if (any(difforder == 0)){
        difforder <- NULL
    }

  return(difforder)

}

seas.dum.net <- function(st,difforder,det.type,ff,ff.n,Y,y,allow.det.season){
# Create seasonal dummies for networks

  if ((if(ff.n > 1){TRUE}else{!any(difforder == max(ff))})
      & frequency(y)>1 & st$season.exist==TRUE & allow.det.season==TRUE){
    sdummy <- TRUE
    # Set type of seasonal dummies
    if (det.type == "auto"){
      if (ff.n == 1 && ff[1] <= 12){
        det.type <- "bin"
      } else {
        det.type <- "trg"
      }
    }
    Xd <- vector("list",ff.n)
    for (s in 1:ff.n){
      if (det.type=="trg"){

        # There was a problem when the fractional seasonalities were < 3, so this is now separated
        Xd[[s]] <- tsutils::seasdummy(length(Y),y=ts(Y,end=end(y),frequency=ff[s]),type="trg",full=TRUE)
        Xd[[s]] <- Xd[[s]][,1:min(length(Xd[[s]][1,]),2)]
      } else {
        Xd[[s]] <- tsutils::seasdummy(length(Y),y=ts(Y,end=end(y),frequency=ff[s]),type="bin")
      }
      colnames(Xd[[s]]) <- paste0("D",s,".",1:length(Xd[[s]][1,]))
    }
    # Xd <- do.call(cbind,Xd)
    # X <- cbind(X,Xd)
  } else {
    sdummy <- FALSE
    Xd <- NULL
  }

  return(list("Xd"=Xd,"det.type"=det.type,"sdummy"=sdummy))

}

create.inputs <- function(y.sc,xreg.sc,lags,xreg.lags,n){
  # Prepare inputs & target
  if (length(lags)>0){
    ylags <- max(lags)
  } else {
    ylags <- 0
  }
  if (!is.null(xreg.sc)){
    xlags <- unlist(lapply(xreg.lags,function(x){if(length(x)){max(x)}else{0}}))
    lag.max <- max(c(ylags,xlags))
  } else {
    lag.max <- ylags
  }
  # Univariate
  if (all(ylags != 0)){
    y.sc.lag <- tsutils::lagmatrix(y.sc,unique(c(0,lags)))
    Y <- y.sc.lag[(lag.max+1):n,1,drop=FALSE]
    colnames(Y) <- "Y"
    X <- y.sc.lag[(lag.max+1):n,2:(length(lags)+1),drop=FALSE]
    colnames(X) <- paste0("X",lags)
  } else {
    Y <- matrix(y.sc[(lag.max+1):n],ncol=1)
    colnames(Y) <- "Y"
    X <- NULL
  }
  # Exogenous
  if (!is.null(xreg.sc)){
    x.p <- dim(xreg.sc)[2]
    x.n <- dim(xreg.sc)[1]
    Xreg <- vector("list",x.p)
    for (i in 1:x.p){
      if (length(xreg.lags[[i]]>0)){
        Xreg[[i]] <- tsutils::lagmatrix(xreg.sc[,i],xreg.lags[[i]])[(lag.max+1):x.n,,drop=FALSE]
        colnames(Xreg[[i]]) <- paste0("Xreg.",i,".",xreg.lags[[i]])
      } else {
        Xreg[[i]] <- NULL
      }
    }
    x.n.all <- lapply(Xreg,function(x){length(x[,1])})
    x.n.all <- x.n.all[x.n.all>0]
    if (any(x.n.all < length(Y))){
      stop("Length of xreg after construction of lags smaller than training sample.")
    }
  } else {
    Xreg <- NULL
  }

  return(list("Y"=Y,"X"=X,"Xreg"=Xreg,"lag.max"=lag.max))

}

auto.hd.elm <- function(Y,X,frm){

  # Use ELM to find hidden nodes
  sz.elm <- max(min(dim(X)[2]+2,length(Y)-2))
  reps.elm <- 20
  # sz.elm <- min(40,max(1,length(Y)-2))

  net <- neuralnet::neuralnet(frm,cbind(Y,X),hidden=sz.elm,threshold=10^10,rep=reps.elm,err.fct="sse",linear.output=FALSE)
  hd.elm <- vector("numeric",reps.elm)
  for (r in 1:reps.elm){
    Z <- as.matrix(tail(neuralnet::compute(net,X,r)$neurons,1)[[1]][,2:(sz.elm+1)])

    type <- "step"
    # Calculate regression
    switch(type,
           "lasso" = {
             fit <- suppressWarnings(glmnet::cv.glmnet(Z,cbind(Y)))
             cf <- as.vector(coef(fit))
             hd.elm[r] <- sum(cf != 0)-1 # -1 for intercept
           },
           {
             reg.data <- as.data.frame(cbind(Y,Z))
             colnames(reg.data) <- c("Y",paste0("X",1:sz.elm))
             # Take care of linear dependency
             alias.fit <- alias(Y~.,data=reg.data)
             alias.x <- rownames(alias.fit$Complete)
             frm.elm <- as.formula(paste0("Y~",paste0(setdiff(colnames(reg.data)[2:(sz.elm+1)],alias.x),collapse="+")))
             fit <- suppressWarnings(lm(frm.elm,reg.data))
             if (type == "step"){
               fit <- suppressWarnings(MASS::stepAIC(fit,trace=0,direction="backward"))
             }
             hd.elm[r] <- sum(summary(fit)$coefficients[,4]<0.05,na.rm=TRUE)-(summary(fit)$coefficients[1,4]<0.05)
           })

  }
  hd <- round(median(hd.elm))
  if (hd<1){
    hd <- 1
  }

  return(hd)

}

auto.hd.cv <- function(Y,X,frm,comb,reps,type=c("cv","valid"),hd.max=NULL){
  # Find number of hidden nodes with CV

  # Setup
  type <- type[1]
  K <- 5                                            # Number of folds
  val.size <- 0.2                                   # Size of validation set
  reps <- min(c(20,max(c(2,reps))))                 # Number of NN reps, maximum 20
  if (is.null(hd.max)){
    hd.max <- max(2,min(dim(X)[2]+2,length(Y)-2)) # Maximum number of hidden nodes
  }

  # Setup folds or validation set
  n <- length(Y)
  if (type == "cv"){
    # Create folds
    if (K >= n){
      stop("Too few observations to perform cross-validation for specification of hidden nodes.")
    }
    # Create fold indices
    idx.all <- sample(1:n)
    cv.cut <- seq(0,n,length.out=K+1)
    idx <- vector("list",K)
    for (i in 1:K){
      idx[[i]] <- which((idx.all <= cv.cut[i+1]) & (idx.all > cv.cut[i]))
    }
  } else {
    # Validation set
    K <- 1
    idx <- list(sample(1:n,round(val.size*n)))
  }

  # Now run CV (1-step ahead)
  err.h <- array(NA,c(hd.max,1),dimnames=list(paste0("H.",1:hd.max),"MSE"))
  for (h in 1:hd.max){
    # For each fold
    err.cv <- vector("numeric",K)
    for (i in 1:K){
      Y.trn <- Y[setdiff(1:n,idx[[i]]),,drop=FALSE]
      X.trn <- X[setdiff(1:n,idx[[i]]),,drop=FALSE]
      Y.tst <- Y[idx[[i]],,drop=FALSE]
      X.tst <- X[idx[[i]],,drop=FALSE]
      net <- neuralnet::neuralnet(frm,cbind(Y.trn,X.trn),hidden=h,rep=reps,err.fct="sse",linear.output=TRUE)
      reps <- length(net$weights) # In case some network is untrained

      # For each training repetition
      frc <- array(NA,c(length(Y.tst),reps))
      for (r in 1:reps){
        frc[,r] <- neuralnet::compute(net,X.tst,r)$net.result
      }
      frc <- frc.comb(frc,comb)
      err.cv[i] <- mean((Y.tst - frc)^2)
    }
    err.h[h] <- mean(err.cv)
  }
  hd <- which(err.h == min(err.h))[1]

  return(list("hd"=hd,"mseH"=err.h))

}

frc.comb <- function(Yhat,comb,na.rm=c(FALSE,TRUE)){
  # Combine forecasts
  na.rm <- na.rm[1]
  r <- dim(Yhat)[2]

  if (r>1){
    switch(comb,
           "median" = {yout <- apply(Yhat,1,median,na.rm=na.rm)},
           "mean" = {yout <- apply(Yhat,1,mean,na.rm=na.rm)},
           "mode" = {# Remove NAs
                     Ytemp <- Yhat
                     Ytemp <- Ytemp[, colSums(is.na(Ytemp))==0]
                     # Calculate only for non-constants
                     idx <- !apply(Ytemp,1,forecast::is.constant)
                     k <- sum(idx)
                     yout <- Yhat[,1]
                     if (k>0){
                       Ycomb <- Ytemp[idx,]
                       if (k>1){
                         Ycomb <- sapply(apply(Ycomb,1,kdemode),function(x){x[[1]][1]})
                       } else {
                         Ycomb <- kdemode(Ycomb)$mode
                       }
                       yout[idx] <- Ycomb
                     }
                    })
  } else {
    yout <- Yhat[,1]
  }

  return (yout)
}
