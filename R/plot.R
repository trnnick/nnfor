# Plotting.
#
# Conventions shared by every plot in the package:
#   * Graphical parameters set by the user (par(), layout(), palettes,
#     colours, fonts) are respected. Nothing calls par() or opens a device,
#     and default line colours come from par("fg") / par("col"), so a dark
#     theme set with par(fg = "white", bg = "black") just works.
#   * Arguments in `...` go to the call that creates the plot frame and
#     override the package defaults (main, xlab, ylim, col, lwd, ...).
#   * Each drawn component has its own `*.args` list of graphical parameters
#     (e.g. members.args = list(col = "pink", lty = 2)), merged over defaults.
#   * Time series plots accept add = TRUE to draw on an existing plot.

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

#' Plot MLP network
#'
#' Produces a plot of the MLP network architecture, or of the in-sample fit.
#'
#' @param x MLP network object, produced using \code{\link{mlp}}.
#' @param r Ensemble member to plot (architecture plot only).
#' @param which What to plot: \code{"network"} for the architecture of member \code{r}, or \code{"fit"} for the in-sample fit.
#' @param ... Graphical arguments passed on. For \code{which = "network"} see \code{\link{plot_network}}; for \code{which = "fit"} see \code{\link{plot_fit}}.
#'
#' @return Invisibly returns \code{x}. Function produces a plot.
#' @note Neurons are coloured with \code{"lightgrey"}. Seasonal dummies are coloured with \code{"lightpink"} and xreg with \code{"lightblue"}. All colours can be changed, see \code{\link{plot_network}}.
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}, \code{\link{mlp}}, \code{\link{plot_network}}, \code{\link{plot_fit}}.
#' @keywords mlp thief ts
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers,reps=1)
#'  plot(fit)
#'  plot(fit, which = "fit")
#' }
#' \dontrun{
#' fit <- mlp(AirPassengers, reps = 5)
#' plot(fit)                                  # architecture of the first network
#' plot(fit, r = 3)                           # architecture of the third network
#' plot(fit, which = "fit")                   # in-sample fit: ensemble and each network
#' plot(fit, which = "fit", members = FALSE)  # ensemble fit only
#' # Further arguments customise the plot: see plot_network() and plot_fit()
#' plot(fit, main = "My network", col.neuron = "gold")
#' plot(fit, which = "fit", fitted.args = list(col = "red", lwd = 2))
#' }
#'
#' @export
#' @method plot mlp
plot.mlp <- function(x, r = 1, which = c("network", "fit"), ...) {
  which <- match.arg(which)
  if (which == "fit") plot_fit(x, ...) else plot_network(x, r = r, ...)
  invisible(x)
}

#' @rdname plot.elm
#'
#' @title Plot ELM network
#'
#' @description Produces a plot of the ELM network architecture, or of the in-sample fit.
#'
#' @param x ELM network object, produced using \code{\link{elm}}.
#' @param r Ensemble member to plot (architecture plot only).
#' @param which What to plot: \code{"network"} for the architecture of member \code{r}, or \code{"fit"} for the in-sample fit.
#' @param ... Graphical arguments passed on. For \code{which = "network"} see \code{\link{plot_network}}; for \code{which = "fit"} see \code{\link{plot_fit}}.
#'
#' @return Invisibly returns \code{x}. Function produces a plot.
#' @note Neurons are coloured with \code{"lightgrey"}. Seasonal dummies are coloured with \code{"lightpink"} and xreg with \code{"lightblue"}. Output connections of unused hidden nodes are dashed. All colours can be changed, see \code{\link{plot_network}}.
#' @author Nikolaos Kourentzes, \email{nikolaos@kourentzes.com}
#' @seealso \code{\link{elm}}, \code{\link{mlp}}, \code{\link{plot_network}}, \code{\link{plot_fit}}.
#' @keywords elm thief ts
#' @examples
#' \dontshow{
#'  fit <- elm(AirPassengers,reps=1)
#'  plot(fit)
#' }
#' \dontrun{
#' fit <- elm(AirPassengers, reps = 5, direct = TRUE)
#' plot(fit)              # unused hidden nodes are dashed, direct connections blue
#' plot(fit, r = 2)       # architecture of the second network
#' plot(fit, which = "fit")  # in-sample fit
#' }
#'
#' @export
#' @method plot elm
plot.elm <- function(x, r = 1, which = c("network", "fit"), ...) {
  which <- match.arg(which)
  if (which == "fit") plot_fit(x, ...) else plot_network(x, r = r, ...)
  invisible(x)
}

# Architecture of ensemble member r, independent of how it was trained:
# layer sizes, input names, and ELM-specific output/direct weights.
network_layout <- function(x, r) {
  is.elm <- inherits(x, "elm")
  is.fast <- inherits(x, "elm.fast")
  reps <- n_members(x)
  if (r > reps) {
    stop("Requested training repetition ", r, " with only ", reps, " available.", call. = FALSE)
  }
  if (is.fast) {
    names <- rownames(x$W.in[[r]])[-1]  # first row is the bias
    if (is.null(names)) names <- paste0("X", seq_len(nrow(x$W.in[[r]]) - 1))
    sizes <- c(length(names), x$hd[r], 1)
  } else if (is.elm) {
    backend <- nn_backend("neuralnet")
    names <- backend$input_names(x$net)
    sizes <- backend$layers(x$net, r)
  } else {
    backend <- object_backend(x)
    names <- backend$input_names(x$net)
    sizes <- backend$layers(x$net, r)
  }
  list(sizes = sizes, names = names, is.elm = is.elm, is.fast = is.fast,
       title = if (is.elm) "ELM" else "MLP")
}

#' Plot the architecture of a network
#'
#' Draws the inputs, hidden layers and output of one ensemble member of an
#' \code{\link{mlp}} or \code{\link{elm}} fit. Called by \code{plot(fit)}.
#' The plot is drawn in the current plotting region, respecting any layout
#' or graphical parameters already set by the user.
#'
#' @param x Network object, produced using \code{\link{mlp}} or \code{\link{elm}}.
#' @param r Ensemble member to plot.
#' @param col.neuron Fill colour of hidden and output neurons, and of lag inputs.
#' @param col.xreg Fill colour of exogenous inputs.
#' @param col.season Fill colour of seasonal dummy inputs.
#' @param col.connection Colour of connections and arrows. Defaults to \code{par("fg")}.
#' @param col.border Border colour of neurons. Defaults to \code{par("fg")}.
#' @param col.inactive Colour of ELM output connections from unused hidden neurons.
#' @param col.direct Colour of ELM direct input-output connections.
#' @param arrow.length Length of the arrow heads, as in \code{\link[graphics]{arrows}}.
#' @param axis.args List of arguments passed to \code{\link[graphics]{axis}} for the layer labels, e.g. \code{list(cex.axis = 0.8)}. Use \code{axis.args = NULL} to omit the labels.
#' @param ... Arguments passed to \code{\link[graphics]{plot}} when creating the frame, overriding the defaults (e.g. \code{main}, \code{sub}, \code{cex.main}).
#'
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{plot.mlp}}, \code{\link{plot.elm}}.
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers, reps = 1)
#'  plot_network(fit, main = "", col.neuron = "white")
#' }
#' \dontrun{
#' fit <- mlp(AirPassengers, reps = 1)
#' # Colours of the inputs: lags of y, regressors (xreg) and seasonal dummies
#' plot_network(fit, col.neuron = "white", col.xreg = "skyblue", col.season = "orange")
#' # Without the layer labels, and with a title of your own
#' plot_network(fit, axis.args = NULL, main = "")
#' # Graphical parameters set with par() are respected, e.g. a dark theme
#' op <- par(bg = "grey15", fg = "white", col.main = "white", col.axis = "white")
#' plot_network(fit)
#' par(op)
#' }
#' @export
plot_network <- function(x, r = 1, col.neuron = "lightgrey", col.xreg = "lightblue",
                         col.season = "lightpink", col.connection = par("fg"),
                         col.border = par("fg"), col.inactive = "grey", col.direct = "blue",
                         arrow.length = 0.1, axis.args = list(), ...) {
  if (!inherits(x, c("mlp", "elm"))) {
    stop("Model must be the output of either mlp or elm functions.", call. = FALSE)
  }
  lay <- network_layout(x, r)
  if (is.null(lay$sizes)) {
    return(plot_custom_network(x, lay, col.neuron, col.xreg, col.season, col.connection,
                               col.border, arrow.length, axis.args, ...))
  }
  sizes <- lay$sizes
  layer.n <- length(sizes) - 1                 # weight layers (hidden + output)
  xx <- c(0, seq(0.1, 0.9, length.out = layer.n + 1), 1)
  yy <- lapply(sizes, function(s) seq(0, 1, length.out = s + 2)[-c(1, s + 2)])
  rd <- min(max(0.015, 1 / ((max(sizes) + 2) * 1.75)), 0.06)  # neuron radius

  inputs.col <- rep(col.neuron, sizes[1])
  inputs.col[grepl("^Xreg\\.", lay$names)] <- col.xreg
  inputs.col[grepl("^D[0-9]+\\.", lay$names)] <- col.season

  do.call(plot, merge_args(list(x = NA, y = NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "",
                                xaxt = "n", yaxt = "n", bty = "n", main = lay$title),
                           list(...)))

  connect <- function(k, i, j, ...) {
    lines(c(xx[k + 1] + rd, xx[k + 2] - rd), c(yy[[k]][i], yy[[k + 1]][j]), ...)
  }

  # Connections. For an ELM the trained output layer is drawn separately.
  for (k in seq_len(layer.n - lay$is.elm)) {
    for (i in seq_len(sizes[k])) for (j in seq_len(sizes[k + 1])) {
      connect(k, i, j, col = col.connection)
    }
  }
  hidden.border <- rep(col.border, sizes[layer.n])
  if (lay$is.elm) {
    active <- rep(TRUE, sizes[layer.n])
    if (!lay$is.fast) active <- as.vector(x$W[[r]] != 0)  # fast ELMs drop unused nodes
    hidden.border[!active] <- col.inactive
    for (i in seq_len(sizes[layer.n])) {
      connect(layer.n, i, 1, col = if (active[i]) col.connection else col.inactive,
              lty = if (active[i]) 1 else 2)
    }
    if (isTRUE(x$direct)) {
      wd <- x$W.dct[[r]]
      for (i in which(wd != 0)) {
        lines(c(xx[2] + rd, xx[layer.n + 2] - rd), c(yy[[1]][i], yy[[layer.n + 1]][1]),
              col = col.direct)
      }
    }
  }

  # Neurons
  for (k in seq_along(sizes)) {
    for (i in seq_len(sizes[k])) {
      fill <- if (k == 1) inputs.col[i] else col.neuron
      border <- if (lay$is.elm && k == layer.n) hidden.border[i] else col.border
      plotrix::draw.circle(xx[k + 1], yy[[k]][i], rd, col = fill, border = border)
    }
  }

  # Input and output arrows
  for (i in seq_len(sizes[1])) {
    arrows(0, yy[[1]][i], xx[2] - rd, length = arrow.length, code = 2, col = col.connection)
  }
  for (i in seq_len(sizes[layer.n + 1])) {
    arrows(xx[layer.n + 2] + rd, yy[[layer.n + 1]][i], 1, length = arrow.length, code = 2,
           col = col.connection)
  }

  # Layer labels
  if (!is.null(axis.args)) {
    inp <- if (sizes[1] > 1) paste0("Inputs\n(", sizes[1], ")") else "Input"
    hid <- if (layer.n > 2) paste0("Hidden ", 1:(layer.n - 1)) else "Hidden"
    hid <- paste0(hid, "\n(", sizes[2:layer.n], ")")
    do.call(axis, merge_args(list(side = 1, at = xx[2:(layer.n + 2)],
                                  labels = c(inp, hid, "Output"), col = NA),
                             axis.args))
  }

  invisible(x)
}

# Inputs -> custom hidden block (printed) -> output, for torch architectures
# given as a function.
plot_custom_network <- function(x, lay, col.neuron, col.xreg, col.season, col.connection,
                                col.border, arrow.length, axis.args, ...) {
  n <- length(lay$names)
  do.call(plot, merge_args(list(x = NA, y = NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "",
                                xaxt = "n", yaxt = "n", bty = "n", main = lay$title),
                           list(...)))
  yy <- seq(0, 1, length.out = n + 2)[-c(1, n + 2)]
  rd <- min(max(0.015, 1 / ((n + 2) * 1.75)), 0.06)
  cols <- rep(col.neuron, n)
  cols[grepl("^Xreg\\.", lay$names)] <- col.xreg
  cols[grepl("^D[0-9]+\\.", lay$names)] <- col.season
  for (i in seq_len(n)) {
    lines(c(0.1 + rd, 0.3), c(yy[i], 0.5), col = col.connection)
    arrows(0, yy[i], 0.1 - rd, length = arrow.length, col = col.connection)
    plotrix::draw.circle(0.1, yy[i], rd, col = cols[i], border = col.border)
  }
  rect(0.3, 0.2, 0.75, 0.8, col = col.neuron, border = col.border)
  # torch prints modules with cli, which uses Unicode rules and bullets when
  # the session supports them; devices such as pdf() cannot draw those. Ask
  # cli for ASCII and replace anything that is still not ASCII.
  old <- options(cli.unicode = FALSE)
  txt <- utils::capture.output(print(torch_member_module(x$net$builder, x$net$n.inputs)[[1]]))
  options(old)
  txt <- gsub("[^ -~]", "?", enc2utf8(txt), perl = TRUE)
  txt <- txt[nzchar(trimws(txt))]
  if (length(txt) > 12) txt <- c(txt[1:11], "...")
  txt <- ifelse(nchar(txt) > 44, paste0(substr(txt, 1, 42), "..."), txt)
  text(0.31, 0.78, paste(txt, collapse = "\n"), adj = c(0, 1), cex = 0.6, family = "mono")
  lines(c(0.75, 0.9 - rd), c(0.5, 0.5), col = col.connection)
  plotrix::draw.circle(0.9, 0.5, rd, col = col.neuron, border = col.border)
  arrows(0.9 + rd, 0.5, 1, length = arrow.length, col = col.connection)
  if (!is.null(axis.args)) {
    do.call(axis, merge_args(list(side = 1, at = c(0.1, 0.525, 0.9), col = NA,
                                  labels = c(paste0("Inputs\n(", n, ")"), "Custom hidden\nlayers", "Output")),
                             axis.args))
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# In-sample fit
# ---------------------------------------------------------------------------

# Default y-limits covering the parts of the given series visible within
# xlim (all of them when xlim is NULL).
visible_ylim <- function(series, xlim = NULL) {
  vals <- unlist(lapply(series, function(s) {
    if (is.null(s)) return(NULL)
    v <- as.matrix(s)
    if (!is.null(xlim)) {
      tt <- as.vector(time(s))
      v <- v[tt >= min(xlim) & tt <= max(xlim), , drop = FALSE]
    }
    v
  }))
  if (!any(is.finite(vals))) return(visible_ylim(series))
  range(vals, na.rm = TRUE)
}

#' Plot the in-sample fit of a network
#'
#' Plots the series together with the ensemble fit and, optionally, the fit
#' of each ensemble member. This is the plot produced by \code{outplot = TRUE}
#' in \code{\link{mlp}} and \code{\link{elm}}, and by
#' \code{plot(fit, which = "fit")}.
#'
#' @param object Network object, produced using \code{\link{mlp}} or \code{\link{elm}}.
#' @param members Logical, draw the fit of each ensemble member.
#' @param add Logical, add to the current plot instead of starting a new one. When \code{TRUE} the series itself is not redrawn.
#' @param members.args List of graphical parameters for the member fits (default \code{list(col = "grey")}).
#' @param fitted.args List of graphical parameters for the ensemble fit (default \code{list(col = "blue")}).
#' @param ... Arguments passed to \code{\link[graphics]{plot}} when drawing the series, overriding the defaults (e.g. \code{main}, \code{ylim}, \code{col}, \code{lwd}).
#'
#' @return Invisibly returns \code{object}.
#' @note Members are only available for models fitted with nnfor 0.9.9 or later.
#' @seealso \code{\link{plot.mlp}}, \code{\link{plot.elm}}.
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers, reps = 2)
#'  plot_fit(fit, main = "Fit", fitted.args = list(col = "red", lwd = 2))
#' }
#' \dontrun{
#' fit <- mlp(AirPassengers, reps = 5)
#' # Style of the network fits (members) and of the ensemble fit
#' plot_fit(fit, members.args = list(col = "pink"), fitted.args = list(col = "red", lwd = 2))
#' # Zoom in with arguments for plot(), e.g. xlim
#' plot_fit(fit, xlim = c(1958, 1961))
#' # Add the fit to a plot of your own
#' plot(AirPassengers, col = "grey40", main = "Air passengers")
#' plot_fit(fit, add = TRUE, members = FALSE)
#' }
#' @export
plot_fit <- function(object, members = TRUE, add = FALSE, members.args = list(),
                     fitted.args = list(), ...) {
  y <- object$y
  all <- object$fitted.all
  members <- members && !is.null(all) && NCOL(all) > 1
  if (!add) {
    yl <- visible_ylim(list(y, object$fitted, if (members) all), list(...)[["xlim"]])
    do.call(plot, merge_args(list(x = y, ylim = yl, xlab = "", ylab = "",
                                  main = paste(toupper(class(object)[1]), "fit")),
                             list(...)))
  }
  if (members) {
    m.args <- merge_args(list(col = "grey"), members.args)
    for (i in seq_len(ncol(all))) do.call(lines, c(list(all[, i]), m.args))
  }
  do.call(lines, c(list(object$fitted), merge_args(list(col = "blue"), fitted.args)))
  invisible(object)
}

# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

#' Plot forecasts of a network
#'
#' Plots the series, the ensemble forecast and, optionally, the forecasts of
#' each ensemble member and the in-sample fit.
#'
#' @param x Object of class \code{forecast.net}, produced by \code{forecast} on an \code{\link{mlp}} or \code{\link{elm}} fit.
#' @param members Logical, draw the forecast of each ensemble member.
#' @param fitted Logical, draw the in-sample fit.
#' @param add Logical, add to the current plot instead of starting a new one. When \code{TRUE} the series itself is not redrawn.
#' @param members.args List of graphical parameters for member forecasts (default \code{list(col = "grey")}).
#' @param mean.args List of graphical parameters for the ensemble forecast (default \code{list(col = "blue", lwd = 2)}).
#' @param fitted.args List of graphical parameters for the in-sample fit (default \code{list(col = "blue", lty = 2)}).
#' @param ... Arguments passed to \code{\link[graphics]{plot}} when drawing the series, overriding the defaults (e.g. \code{main}, \code{xlim}, \code{ylim}, \code{col}).
#'
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{forecast.mlp}}, \code{\link{forecast.elm}}.
#' @examples
#' \dontshow{
#'  fit <- mlp(AirPassengers, reps = 2)
#'  frc <- forecast(fit, h = 12)
#'  plot(frc, members = FALSE, main = "", mean.args = list(col = "red"))
#' }
#' \dontrun{
#' fit <- mlp(AirPassengers, reps = 5)
#' frc <- forecast(fit, h = 24)
#' plot(frc)                                # forecasts of the ensemble and of each network
#' plot(frc, members = FALSE, fitted = TRUE)  # ensemble only, with the in-sample fit
#' plot(frc, xlim = c(1958, 1963), mean.args = list(col = "red"))  # zoom, custom style
#' # Add the forecasts to a plot of your own
#' plot(AirPassengers, xlim = c(1949, 1963), ylim = c(100, 750))
#' plot(frc, add = TRUE, members = FALSE)
#' }
#' @export
#' @method plot forecast.net
plot.forecast.net <- function(x, members = TRUE, fitted = FALSE, add = FALSE,
                              members.args = list(), mean.args = list(),
                              fitted.args = list(), ...) {
  method <- if (any(x$method == "elm")) "elm" else x$method[1]
  all <- x$all.mean
  members <- members && NCOL(all) > 0
  if (!add) {
    xl <- list(...)[["xlim"]] %||% range(time(x$x), time(x$mean))
    yl <- visible_ylim(list(x$x, x$mean, if (members) all, if (fitted) x$fitted), xl)
    do.call(plot, merge_args(list(x = x$x, xlim = xl, ylim = yl, xlab = "", ylab = "",
                                  main = paste("Forecasts from", toupper(method))),
                             list(...)))
  }
  if (fitted) {
    do.call(lines, c(list(x$fitted), merge_args(list(col = "blue", lty = 2), fitted.args)))
  }
  # With a single-step horizon lines are invisible, so use markers instead.
  single <- length(x$mean) == 1
  draw <- function(z, args, pch) {
    if (single) {
      do.call(points, c(list(time(x$mean), as.vector(z)), merge_args(list(pch = pch), args)))
    } else {
      do.call(lines, c(list(z), args))
    }
  }
  if (members) {
    m.args <- merge_args(list(col = "grey"), members.args)
    for (i in seq_len(NCOL(all))) draw(all[, i], m.args, pch = 1)
  }
  draw(x$mean, merge_args(list(col = "blue", lwd = 2), mean.args), pch = 20)
  invisible(x)
}

# ---------------------------------------------------------------------------
# elm.fast
# ---------------------------------------------------------------------------

#' Plot ELM (fast) fit
#'
#' Scatter plot of actual against fitted values of an \code{\link{elm.fast}} network.
#'
#' @param x Object of class \code{elm.fast}.
#' @param members Logical, draw the fitted values of each ensemble member.
#' @param members.args List of graphical parameters for member points (default \code{list(col = "gray30", pch = ".")}).
#' @param fitted.args List of graphical parameters for the ensemble points (default \code{list(pch = 20)}).
#' @param line.args List of graphical parameters for the diagonal reference line (default \code{list(col = "grey")}); \code{NULL} to omit it.
#' @param ... Arguments passed to \code{\link[graphics]{plot}} when creating the frame, overriding the defaults.
#'
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{elm.fast}}.
#' @examples
#' \dontrun{
#' x <- matrix(rnorm(300), 100, 3)
#' y <- x %*% c(1, -1, 0.5) + rnorm(100, sd = 0.3)
#' fit <- elm.fast(y, x, reps = 5)
#' plot(fit)                                     # fitted against actual values
#' plot(fit, members = FALSE, line.args = NULL)  # ensemble only, no reference line
#' }
#' @export
#' @method plot elm.fast
plot.elm.fast <- function(x, members = TRUE, members.args = list(), fitted.args = list(),
                          line.args = list(), ...) {
  fitted <- x$fitted %||% frc.comb(x$fitted.all, x$comb)
  lim <- range(x$y, fitted, x$fitted.all)
  lim <- lim + c(-1, 1) * 0.04 * diff(lim)
  do.call(plot, merge_args(list(x = NA, y = NA, xlab = "Fitted", ylab = "Actual",
                                xlim = lim, ylim = lim),
                           list(...)))
  if (!is.null(line.args)) {
    do.call(lines, c(list(lim + c(-1, 1) * 0.2 * diff(lim), lim + c(-1, 1) * 0.2 * diff(lim)),
                     merge_args(list(col = "grey"), line.args)))
  }
  if (members) {
    m.args <- merge_args(list(col = "gray30", pch = "."), members.args)
    for (r in seq_len(ncol(x$fitted.all))) {
      do.call(points, c(list(x$fitted.all[, r], as.vector(x$y)), m.args))
    }
  }
  do.call(points, c(list(as.vector(fitted), as.vector(x$y)), merge_args(list(pch = 20), fitted.args)))
  invisible(x)
}
