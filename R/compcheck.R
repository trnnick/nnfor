seasoncheck <- function(y,m=NULL,alpha=0.05,decomposition=c("multiplicative","additive"),cma){
  # Check presence of seasonality
  #
  # Inputs:
  #   y             Time series vector (can be ts object)
  #   m             Seasonal period. If y is a ts object then the default is its frequency.
  #   alpha         Significance level for statistical tests (kpss and friedman)
  #   decomposition Type of seasonal decomposition: "multiplicative" or "additive".
  #   cma           Input precalculated level/trend for the analysis. Overrides trend=NULL.

  decomposition <- match.arg(decomposition,c("multiplicative","additive"))
  n <- length(y)

  # Get m (seasonality)
  if (is.null(m)){
    if (any(class(y) == "ts")){
      m <- frequency(y)
    } else {
      stop("Seasonality not defined (y not ts object).")
    }
  }

  # Get starting period of season
  if (any(class(y) == "ts")){
    s <- start(y)
    s <- s[2]
    # Temporal aggregation can mess-up s, so override if needed
    if (is.na(s)){s<-1}
  } else {
    s <- 1
  }

  # Check if multiplicative decomposition is possible
  if ((decomposition == "multiplicative") && (min(y)<=0)){
    decomposition <- "additive"
  }

  # Check if provided CMA is of appropriate length
  if (!is.null(cma)){
    if (length(cma) != n){
      cma <- NULL
    }
  }

  # Calculate CMA if needed
  if (is.null(cma)){
    cma <- tsutils::cmav(y=y,ma=m,fill=TRUE,outplot=FALSE)
  }

  # Remove trend
  if (decomposition == "multiplicative"){
    ynt <- y/cma
  } else {
    ynt <- y-cma
  }

  # Reshape series in seasonal matrix
  k <- m - (n %% m)
  ks <- s-1
  ke <- m - ((n+ks) %% m)
  ynt <- c(rep(NA,times=ks),as.vector(ynt),rep(NA,times=ke))
  ns <- length(ynt)/m
  ynt <- matrix(ynt,nrow=ns,ncol=m,byrow=TRUE)
  rownames(ynt) <- paste("s",1:ns,sep="")

  # Check seasonality with Friedman
  if (m>1 && (length(y)/m)>=2){
    season.pval <- friedman.test(ynt)$p.value
    season.exist <- season.pval <= alpha
  } else {
    season.pval <- NULL
    season.exist <- NULL
  }

  return(list(season.exist=season.exist,season.pval=season.pval,cma=cma))

}

trendcheck <- function(y){
  forecast::ets(y,model="ANN")$aic > forecast::ets(y,model="AAN")$aic
}
