#' @export
#' @method plot forecast.net

plot.forecast.net <- function(x,...){
  # Plot function for NNs
  method <- x$method
  if (any(method=="elm.fast")){
    method <- "ELM"
  }
  reps <- dim(x$all.mean)[2]
  ts.plot(x$x,x$all.mean,x$mean,
          col=c("black",rep("grey",reps),"blue"),lwd=c(1,rep(1,reps),2),
          main=paste("Forecasts from",toupper(method)))
  # If h==1 then use markers
  if (length(x$mean)==1){
    points(rep(time(x$mean),reps+1),c(x$all.mean,x$mean),
           pch=c(rep(1,reps),20),col=c(rep("grey",reps),"blue"))
  }


}
