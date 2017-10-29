kdemode <- function(data,type=c("SJ","nrd0")){
# Return mode of a vector as calculated using KDE
#
# Inputs:
#   data      One-dimensional vector of data
#   type      Bandwidth selection:
#               - SJ: Sheater and Jones method
#               - nrd0: Silverman heuristic
#
# Outputs:
#   mode      Estimated mode value.
#
# Example:
#   data <- rnorm(200,mean=0,sd=1)
#   kdemode(data)
#
# Notes:
#   For a discussion of the selection between mean, median and mode
#   for the combination of forecasts see:
#   Kourentzes, N., Barrow, D. K., & Crone, S. F. (2014).
#   Neural network ensemble operators for time series forecasting.
#   Expert Systems with Applications, Volume 41, Issue 9, Pages 4235-4244
#
# Nikolaos Kourentzes, 2017 <nikolaos@kourentzes.com>

  # Defaults
  type <- match.arg(type,c("SJ","nrd0"))

  # Fix from/to
  from <- min(data)-0.1*diff(range(data))
  to <- max(data)+0.1*diff(range(data))

  # Calculate KDE
  ks <- density(data,bw="SJ",n=512,from=from,to=to)
  x <- ks$x
  f <- ks$y
  h <- ks$bw

  # Find mode
  mo <- x[which(f==max(f))][1] # mode

 return(list(mode=mo,xd=x,fd=f,h=h))

}
