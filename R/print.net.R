#' @export
#' @method print mlp

print.mlp <- function(x, ...){
  print.net(x,...)
}

#' @export
#' @method print elm

print.elm <- function(x, ...){
  print.net(x,...)
}

print.net <- function(x, ...){

  is.elm.fast <- any(class(x)=="elm.fast")
  difforder <- x$difforder
  sdummy <- x$sdummy
  d <- length(difforder)
  if (is.elm.fast){
    reps <- length(x$b)
  } else {
    reps <- length(x$net$weights)
  }
  hd <- x$hd
  xreg.lags <- x$xreg.lags

  if (length(hd)>1 & !is.elm.fast){
    hdt <- paste0(hd,",",collapse="")
    hdt <- paste0("(", substr(hdt,1,nchar(hdt)-1) ,")")
    hde <- "s"
  } else {
    hdt <- hd
    if (is.elm.fast){
      hdt <- paste0(min(hdt)," up to ",max(hdt))
    }
    if (any(hd>1)){
      hde <- "s"
    } else {
      hde <- ""
    }
  }

  dtx <- ""
  if (any(class(x)=="elm")){
    if (x$direct == TRUE){
      dtx <- ", direct output connections"
    }
  }

  method <- class(x)
  if (any(method == "elm")){
    method <- "elm"
  }
  if (is.elm.fast){
    fst <- " (fast)"
  } else {
    fst <- ""
  }

  writeLines(paste0(toupper(method),fst," fit with ", hdt," hidden node", hde, dtx," and ", reps, " repetition",if(reps>1){"s"},"."))
  if (d>0){
    writeLines(paste0("Series modelled in differences: ", paste0("D",difforder,collapse=""), "."))
  }
  if (all(x$lags != 0)){
    writeLines(paste0("Univariate lags: (",paste0(x$lags,collapse=","),")"))
  }
  if (!is.null(xreg.lags)){
    null.xreg <- lapply(xreg.lags,length)==0
    p <- length(xreg.lags) - sum(null.xreg)
    if (p > 0){
      if (p == 1){
        rge <- ""
      } else {
        rge <- "s"
      }
      writeLines(paste0(p," regressor",rge," included."))
      pi <- 1
      for (i in which(!null.xreg)){
        writeLines(paste0("- Regressor ",pi," lags: (",paste0(xreg.lags[[i]],collapse=","),")"))
        pi <- pi + 1
      }
    }
  }
  if (sdummy == TRUE){
    writeLines(paste0("Deterministic seasonal dummies included."))
  }
  if (reps>1){
    writeLines(paste0("Forecast combined using the ", x$comb, " operator."))
  }
  if (any(class(x)=="elm")){
    writeLines(paste0("Output weight estimation using: ", x$type, "."))
  }
  writeLines(paste0("MSE: ",round(x$MSE,4),"."))

}
