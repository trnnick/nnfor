# Map of the inner workings of the package.
#
# The map is declarative: each box records the function it represents (`ref`)
# and the function that calls it (`caller`). tests/testthat/test-map.R checks
# that every `caller` really calls its `ref`, so the map cannot silently drift
# away from the code.

#' Map of how information flows through nnfor
#'
#' Builds a map of the steps that \code{\link{mlp}} and \code{\link{elm}} go
#' through when fitting a network, or that \code{forecast} goes through when
#' producing forecasts. Each box is an internal or exported function of the
#' package, showing what it does and what it passes on. Plot the map with
#' \code{plot}, or print it for a text outline.
#'
#' When \code{x} is a fitted network, the map is annotated with the values
#' used by that fit (seasonal periods, differencing, lags, dummies, hidden
#' nodes, ...) and steps that the fit did not use are greyed out.
#'
#' @param x Either \code{"mlp"} or \code{"elm"}, or a network fitted with \code{\link{mlp}} or \code{\link{elm}}.
#' @param stage \code{"fit"} for model fitting or \code{"forecast"} for forecasting.
#'
#' @return An object of class \code{nnfor_map}: a list with \code{nodes} (one
#'   row per step: function, description, output, stage and the condition
#'   under which it runs) and \code{edges} (flows between steps), plus
#'   information used for plotting.
#' @seealso \code{\link{plot.nnfor_map}}, \code{\link{mlp}}, \code{\link{elm}}.
#' @examples
#' plot(nnfor_map("mlp"))
#' print(nnfor_map("elm", stage = "forecast"))
#' \dontrun{
#'  fit <- mlp(AirPassengers)
#'  plot(nnfor_map(fit))
#'  plot(nnfor_map(fit, stage = "forecast"))
#' }
#' @export
nnfor_map <- function(x = c("mlp", "elm"), stage = c("fit", "forecast")) {
  stage <- match.arg(stage)
  fit <- NULL
  if (inherits(x, c("mlp", "elm"))) {
    fit <- x
    model <- if (inherits(x, "elm")) "elm" else "mlp"
  } else {
    model <- match.arg(x)
  }
  map <- if (stage == "fit") map_fit(model) else map_forecast(model)
  map$model <- model
  map$stage <- stage
  map$nodes$value <- NA_character_
  map$nodes$inactive <- FALSE
  if (!is.null(fit)) map <- annotate_map(map, fit)
  structure(map, class = "nnfor_map")
}

# ---------------------------------------------------------------------------
# Map definitions
# ---------------------------------------------------------------------------

# One box: `col`/`row` give its position, `ref` the function it stands for,
# `caller` the function calling it (NA for entry points), `cond` the
# condition under which it runs (NA if always).
map_node <- function(id, label, desc, out, col, row, ref = NA, caller = NA, cond = NA) {
  data.frame(id = id, label = label, desc = desc, out = out, col = col, row = row,
             ref = ref, caller = caller, cond = cond, stringsAsFactors = FALSE)
}

# Flows between boxes. type: "data", "spec" (stored settings or
# transformation details needed later) or "loop" (repeated step).
map_edges <- function(...) {
  e <- do.call(rbind, lapply(list(...), function(x) {
    data.frame(from = x[1], to = x[2], type = if (length(x) > 2) x[3] else "data",
               stringsAsFactors = FALSE)
  }))
  e
}

map_fit <- function(model) {
  fun <- paste0(model, "()")
  shared <- rbind(
    map_node("call", fun, "arguments y, m, xreg, lags, hd, reps, ...", "", 1, 1, ref = model),
    map_node("model", "model = <fit>", "previously fitted network", "", 1, 3,
             cond = "if a model is reused"),
    map_node("periods", "resolve_periods()", "seasonal periods from m, else from y", "ff", 2, 1,
             "resolve_periods", model),
    map_node("lags", "default_lags()", "default lags for y and xreg", "lags, xreg.lags, keep", 2, 2,
             "default_lags", "prepare_data"),
    map_node("reuse", "reused_spec()", "copy lags, differencing and dummies of the model",
             "fixed specification", 2, 3, "reused_spec", "prepare_data", "if a model is reused"),
    map_node("season", "seasoncheck()", "test for seasonality (Friedman) around the trend",
             "seasonality, trend (cma)", 3, 1, "seasoncheck", "preprocess"),
    map_node("difforder", "resolve_difforder()", "trend and seasonal unit root tests",
             "difforder", 3, 2, "resolve_difforder", "preprocess"),
    map_node("diff", "difference_series()", "apply the differences", "y.d (all stages)", 3, 3,
             "difference_series", "preprocess"),
    map_node("scale", "linscale()", "scale target to [-0.8, 0.8]", "y.sc, minmax", 3, 4,
             "linscale", "preprocess"),
    map_node("sdummy", "use_season_dummies()", "dummies if seasonal and not seasonally differenced",
             "sdummy, det.type", 4, 1, "use_season_dummies", "preprocess"),
    map_node("dummies", "season_dummies_fit()", "binary or trigonometric seasonal dummies",
             "Xd", 4, 2, "season_dummies_fit", "preprocess", "if dummies are used"),
    map_node("xreg", "scale_xreg()", "align xreg with the differenced target, scale",
             "xreg.sc, xreg.minmax", 4, 3, "scale_xreg", "preprocess", "if xreg is given"),
    map_node("inputs", "training_inputs()", "lagged target and xreg via lag_inputs()",
             "Y, X", 4, 4, "training_inputs", "preprocess"),
    map_node("select", "select_inputs()", "stepwise regression (lasso fallback), rebuild inputs",
             "kept lags, xreg lags, dummies", 4, 5, "select_inputs", "preprocess",
             "if sel.lag = TRUE")
  )
  out.col <- if (model == "mlp") 6 else 7
  tail.nodes <- rbind(
    map_node("finalise", "finalise_fit()", "reverse scaling and differencing, combine members (frc.comb)",
             "fitted, fitted.all, MSE", out.col, 1, "finalise_fit", model),
    map_node("object", paste(model, "object"), "network, specification and fit", "", out.col, 2),
    map_node("plot", "plot_fit()", "plot of the in-sample fit", "", out.col, 3, "plot_fit", model,
             "if outplot = TRUE"),
    map_node("forecast", "forecast()", "see nnfor_map(stage = \"forecast\")", "", out.col, 4,
             paste0("forecast.", model))
  )
  edges.shared <- list(
    c("call", "periods"), c("call", "lags"), c("model", "reuse"),
    c("reuse", "lags", "spec"), c("reuse", "difforder", "spec"),
    c("periods", "lags"), c("periods", "season"),
    c("season", "difforder"), c("difforder", "diff"), c("diff", "scale"),
    c("difforder", "sdummy"), c("sdummy", "dummies"), c("difforder", "xreg", "spec"),
    c("lags", "inputs"), c("scale", "inputs"), c("xreg", "inputs"),
    c("inputs", "select"), c("dummies", "select"), c("select", "inputs", "loop"),
    c("diff", "finalise", "spec"),
    c("finalise", "object"), c("object", "plot"), c("object", "forecast")
  )

  if (model == "mlp") {
    train <- rbind(
      map_node("hidden", "select_hidden_cv() / _elm()", "number of hidden nodes by validation, CV or ELM",
               "hd", 5, 1, "select_hidden_cv", "mlp", "if hd = NULL"),
      map_node("train", "backend$train()", "train reps networks with neuralnet or torch", "net", 5, 2,
               "nn_backend", "mlp"),
      map_node("predict", "backend$predict()", "in-sample output of each member", "scaled fit per member",
               5, 3, "predict_members", "mlp")
    )
    edges <- c(edges.shared, list(
      c("select", "hidden"), c("select", "train"), c("hidden", "train"),
      c("train", "predict"), c("predict", "finalise")))
    stages <- c("Arguments", "Specification", "Target", "Network inputs", "Training", "Output")
  } else {
    train <- rbind(
      map_node("random", "elm_random_layer()", "random hidden weights drawn by neuralnet", "net",
               5, 1, "elm_random_layer", "elm", "if barebone = FALSE"),
      map_node("hiddenact", "elm_hidden()", "logistic hidden layer activations", "H", 5, 2,
               "elm_hidden", "elm_fit_output", "if barebone = FALSE"),
      map_node("fast", "elm.fast()", "random weights (init.w), fast.sig activation", "H", 5, 3,
               "elm.fast", "elm", "if barebone = TRUE"),
      map_node("weights", "elm.train()", "output weights by lasso, ridge, step or lm",
               "W, b, W.dct", 6, 1, "elm.train", "elm_fit_output"),
      map_node("predict", "elm_predict_member()", "in-sample output of each member",
               "scaled fit per member", 6, 2, "elm_predict_member", "elm")
    )
    edges <- c(edges.shared, list(
      c("select", "random"), c("select", "fast"), c("random", "hiddenact"),
      c("hiddenact", "weights"), c("fast", "weights"), c("weights", "predict"),
      c("predict", "finalise")))
    stages <- c("Arguments", "Specification", "Target", "Network inputs", "Hidden layer",
                "Output layer", "Output")
  }

  list(nodes = rbind(shared, train, tail.nodes), edges = do.call(map_edges, edges),
       stages = stages, groups = list(list(cols = c(3, 4), label = "preprocess()")),
       title = paste0("How ", fun, " fits a network"))
}

map_forecast <- function(model) {
  predictor <- if (model == "mlp") "backend$predict()" else "elm_predict_member()"
  nodes <- rbind(
    map_node("call", "forecast()", "arguments object, h, y, xreg", "h (default: max(ff))", 1, 1,
             "forecast_network", paste0("forecast.", model)),
    map_node("object", paste(model, "object"), "stored specification and network",
             "lags, difforder, minmax, ...", 1, 2),
    map_node("diff", "difference_series()", "difference y as when fitting", "y.d (all stages)", 2, 1,
             "difference_series", "forecast_network"),
    map_node("scale", "linscale()", "scale with the stored minmax", "history", 2, 2,
             "linscale", "forecast_network"),
    map_node("xreg", "rescale_xreg()", "scale xreg with the stored minmax", "xreg.sc", 2, 3,
             "rescale_xreg", "forecast_network", "if xreg is used"),
    map_node("dummies", "season_dummies_forecast()", "dummies continuing the in-sample pattern",
             "Xd", 2, 4, "season_dummies_forecast", "forecast_network", "if dummies are used"),
    map_node("inputs", "lag_inputs()", "input row from the history and xreg, as in training",
             "input row", 3, 1, "lag_inputs", "forecast_network"),
    map_node("predict", predictor, "next value; appended to the history", "scaled forecast",
             3, 2, "member_predictor", "forecast_network", "for each member and step"),
    map_node("unscale", "linscale()", "reverse the scaling", "", 4, 1, "linscale", "forecast_network"),
    map_node("undiff", "undifference_forecast()", "reverse the differencing recursively", "",
             4, 2, "undifference_forecast", "forecast_network"),
    map_node("comb", "frc.comb()", "combine members: median, mean or mode", "", 4, 3,
             "frc.comb", "forecast_network"),
    map_node("out", "forecast.net object", "point forecasts and member forecasts",
             "mean, all.mean", 5, 1),
    map_node("plot", "plot()", "plot of the forecasts", "", 5, 2, "plot.forecast.net", NA)
  )
  edges <- map_edges(
    c("call", "diff"), c("object", "scale", "spec"), c("object", "xreg", "spec"),
    c("diff", "scale"), c("scale", "inputs"), c("xreg", "inputs"), c("dummies", "predict"),
    c("inputs", "predict"), c("predict", "inputs", "loop"),
    c("predict", "unscale"), c("unscale", "undiff"), c("undiff", "comb"),
    c("diff", "undiff", "spec"), c("comb", "out"), c("out", "plot"))
  list(nodes = nodes, edges = edges,
       stages = c("Arguments", "Prepare", "Recursive forecast", "Post-process", "Output"),
       groups = list(list(cols = 2:4, label = "forecast_network()")),
       title = "How forecast() produces forecasts")
}

# ---------------------------------------------------------------------------
# Annotation with a fitted network
# ---------------------------------------------------------------------------

annotate_map <- function(map, fit) {
  nd <- map$nodes
  set <- function(id, value = NA, inactive = FALSE) {
    i <- nd$id == id
    if (!any(i)) return(invisible())
    if (length(value) == 1 && !is.na(value)) nd$value[i] <<- value
    if (inactive) nd$inactive[i] <<- TRUE
  }
  shorten <- function(v) {
    s <- paste(v, collapse = ",")
    if (nchar(s) > 24) paste0(substr(s, 1, 22), "...") else s
  }
  is.fast <- inherits(fit, "elm.fast")
  has.xreg <- length(fit$xreg.minmax) > 0
  reps <- n_members(fit)
  ff <- fit$ff %||% get_frequencies(fit$y)
  dif <- if (length(fit$difforder)) paste("difforder =", shorten(fit$difforder)) else "no differencing"
  dum <- if (isTRUE(fit$sdummy)) paste0(fit$det.type, " dummies for ", shorten(fit$ff.det)) else "no dummies"
  lags <- paste0("lags = ", if (has_lags(fit$lags)) shorten(fit$lags) else "none",
                 if (has.xreg) paste0("; xreg lags = ", shorten(sapply(fit$xreg.lags, shorten))) else "")

  if (map$stage == "fit") {
    set("periods", paste("ff =", shorten(ff)))
    set("difforder", dif)
    set("diff", dif)
    set("sdummy", dum)
    set("dummies", dum, inactive = !isTRUE(fit$sdummy))
    set("xreg", if (has.xreg) paste(length(fit$xreg.minmax), "regressor(s)"), inactive = !has.xreg)
    set("select", lags)
    if (!is.null(fit$MSEH)) set("hidden", paste("hd =", fit$hd, "of", nrow(fit$MSEH), "tried"))
    set("train", paste0(fit$backend %||% "neuralnet", ": ",
                        if (is.function(fit$hd)) "custom layers" else paste("hd =", shorten(fit$hd)),
                        ", ", reps, " members"))
    set("random", if (!is.fast) paste0("hd = ", shorten(fit$hd), ", ", reps, " members"), inactive = is.fast)
    set("hiddenact", inactive = is.fast)
    set("fast", if (is.fast) paste0("hd = ", min(fit$hd), "-", max(fit$hd), ", ", reps, " members"),
        inactive = !is.fast)
    set("weights", paste0("type = ", fit$type, if (isTRUE(fit$direct)) ", direct" else ""))
    set("finalise", paste0(fit$comb, " of ", reps, ", MSE = ", signif(fit$MSE, 4)))
  } else {
    set("call", paste("default h =", max(ff)))
    set("object", lags)
    set("diff", dif)
    set("xreg", if (has.xreg) paste(length(fit$xreg.minmax), "regressor(s)"), inactive = !has.xreg)
    set("dummies", dum, inactive = !isTRUE(fit$sdummy))
    set("predict", paste(reps, "members"))
    set("comb", paste("comb =", fit$comb))
  }
  map$nodes <- nd
  map
}

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

#' @rdname plot.nnfor_map
#' @export
print.nnfor_map <- function(x, ...) {
  nd <- x$nodes
  writeLines(paste0(x$title, if (any(!is.na(nd$value))) " (annotated with a fitted network)" else "", "\n"))
  for (s in seq_along(x$stages)) {
    sel <- nd[nd$col == s, , drop = FALSE]
    if (!nrow(sel)) next
    writeLines(paste0("[", s, "] ", x$stages[s]))
    for (i in order(sel$row)) {
      n <- sel[i, ]
      writeLines(paste0("    ", formatC(n$label, width = -28), n$desc,
                        if (!is.na(n$cond)) paste0(" [", n$cond, "]") else "",
                        if (n$inactive) " [not used]" else ""))
      detail <- if (!is.na(n$value)) n$value else n$out
      if (nzchar(detail)) writeLines(paste0("    ", strrep(" ", 28), "-> ", detail))
    }
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

#' Plot or print a map of nnfor
#'
#' Draws the map produced by \code{\link{nnfor_map}}. Columns are stages,
#' boxes are functions, with what they produce underneath. Solid arrows pass
#' data, dotted arrows pass stored settings (such as the differencing needed
#' to reverse a transformation) and dashed arrows are repeated steps. Dashed
#' boxes are conditional steps; grey boxes were not used by the annotated fit.
#'
#' The map is drawn in the current plotting region and respects graphical
#' parameters set by the user. Text is sized to the plotting area: one size
#' for all boxes, the largest at which every box fits, with descriptions
#' re-wrapped to the box width. Set \code{cex} to fix the size instead. The
#' map is best viewed on a wide device; in RStudio the plot pane can be
#' small, so saving the map to a file, as in the examples, gives a better
#' view.
#'
#' @param x Object of class \code{nnfor_map}.
#' @param col.stages Colours for the stages (recycled).
#' @param col.text Colour of the text. Defaults to \code{par("fg")}.
#' @param col.value Colour of values from an annotated fit.
#' @param col.inactive Colour of steps not used by an annotated fit.
#' @param col.arrows Colour of the arrows. Defaults to \code{par("fg")}.
#' @param legend Logical, add a legend.
#' @param cex Size of the text in the boxes. \code{NULL} (default) chooses the
#'   largest size at which the text of every box fits, up to \code{cex.max}.
#' @param cex.max Largest text size used when \code{cex = NULL}.
#' @param ... Arguments passed to \code{\link[graphics]{plot}} when creating the frame, overriding the defaults (e.g. \code{main}, \code{cex.main}).
#'
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{nnfor_map}}.
#' @examples
#' plot(nnfor_map("mlp"), main = "")
#' \dontrun{
#'  # Save full-screen maps as images, outside the RStudio plot pane
#'  png("nnfor_map_mlp.png", width = 1920, height = 1080, res = 120)
#'  plot(nnfor_map("mlp"))
#'  dev.off()
#'
#'  # A map annotated with a fitted network
#'  fit <- mlp(AirPassengers)
#'  png("nnfor_map_fit.png", width = 1920, height = 1080, res = 120)
#'  plot(nnfor_map(fit))
#'  dev.off()
#'
#'  # Or as a PDF, which can be zoomed without losing quality
#'  pdf("nnfor_map_forecast.pdf", width = 16, height = 9)
#'  plot(nnfor_map(fit, stage = "forecast"))
#'  dev.off()
#' }
#' @export
#' @method plot nnfor_map
plot.nnfor_map <- function(x, col.stages = c("#0072B2", "#009E73", "#E69F00", "#D55E00",
                                             "#CC79A7", "#56B4E9", "#882255"),
                           col.text = par("fg"), col.value = "#0072B2", col.inactive = "grey60",
                           col.arrows = par("fg"), legend = TRUE, cex = NULL, cex.max = 1.2, ...) {
  nd <- x$nodes
  n.col <- max(nd$col)
  n.row <- max(nd$row)
  bw <- 0.8    # box width (column spacing is 1)
  bh <- 0.82   # box height (row spacing is 1); the gaps still carry the arrows
  pad <- 0.04  # inner margin of a box, in plot units
  spacing <- 1.65  # strheight() covers capitals only, not descenders or leading

  do.call(plot, merge_args(list(x = NA, y = NA, xlim = c(0.5, n.col + 0.5),
                                ylim = c(n.row + if (legend) 0.75 else 0.5, -0.45),
                                xlab = "", ylab = "", axes = FALSE, main = x$title),
                           list(...)))

  # Wrap text to a width (in plot units) at a given size. Text breaks between
  # words; function names (code = TRUE) break after an underscore or before
  # " / ", onto at most two lines so they stay recognisable. Returns NULL if
  # the text cannot fit.
  wrap <- function(txt, width, cex, font, code = FALSE) {
    tokens <- if (code) {
      strsplit(txt, "(?<=[[:alnum:]]_)|(?= / )", perl = TRUE)[[1]]
    } else {
      strsplit(txt, " ", fixed = TRUE)[[1]]
    }
    sep <- if (code) "" else " "
    out <- character(0)
    cur <- ""
    for (w in tokens) {
      cand <- if (nzchar(cur)) paste0(cur, sep, w) else w
      if (!nzchar(cur) || strwidth(cand, cex = cex, font = font) <= width) {
        cur <- cand
      } else {
        out <- c(out, cur)
        cur <- trimws(w)
      }
    }
    out <- trimws(c(out, cur))
    if (code && length(out) > 2) return(NULL)
    out
  }

  # Lines of box i at size `cex`: list(text, font, role); NULL if a function
  # name cannot fit.
  layout_box <- function(i, cex) {
    n <- nd[i, ]
    detail <- if (!is.na(n$value)) n$value else n$out
    parts <- list(list(n$label, 2, "main", TRUE),
                  list(n$desc, 1, "main", FALSE),
                  if (nzchar(detail)) list(paste("->", detail), 3,
                                           if (!is.na(n$value)) "value" else "main", FALSE),
                  if (!is.na(n$cond)) list(n$cond, 3, "cond", FALSE))
    out <- list()
    for (p in parts[!vapply(parts, is.null, logical(1))]) {
      txt <- wrap(p[[1]], bw - 2 * pad, cex, p[[2]], code = p[[4]])
      if (is.null(txt)) return(NULL)
      for (t in txt) out[[length(out) + 1]] <- list(t, p[[2]], p[[3]])
    }
    out
  }
  box_fits <- function(l, cex) {
    if (is.null(l)) return(FALSE)
    w <- max(vapply(l, function(z) strwidth(z[[1]], cex = cex, font = z[[2]]), numeric(1)))
    w <= bw - 2 * pad && length(l) * abs(strheight("Mg", cex = cex)) * spacing <= bh - 2 * pad
  }
  # Largest size at which box i fits
  box_cex <- function(i, upper) {
    cx <- upper
    while (cx > 0.3 && !box_fits(layout_box(i, cx), cx)) cx <- cx * 0.97
    cx
  }

  # One size for all boxes, for a consistent look: the size of the box that
  # needs the smallest text to fit.
  if (is.null(cex)) {
    cex <- min(vapply(seq_len(nrow(nd)), box_cex, numeric(1), upper = cex.max))
  }
  boxes <- lapply(seq_len(nrow(nd)), function(i) {
    layout_box(i, cex) %||% list(list(nd$label[i], 2, "main"))
  })

  # Stage headers and group brackets, sized with the box text
  hcex <- min(1.1, max(0.75, cex))
  stage.col <- rep_len(col.stages, n.col)
  text(seq_len(n.col), 0.12, x$stages, col = stage.col, font = 2, cex = hcex)
  for (g in x$groups) {
    gx <- range(g$cols) + c(-0.45, 0.45)
    lines(gx[c(1, 1, 2, 2)], c(0, -0.22, -0.22, 0), col = col.text)
    text(mean(gx), -0.32, g$label, col = col.text, font = 3, cex = hcex)
  }

  # Arrows first, so boxes sit on top
  draw_map_edges(x$edges, nd, bw, bh, col.arrows)
  lh <- abs(strheight("Mg", cex = cex)) * spacing
  for (i in seq_len(nrow(nd))) {
    n <- nd[i, ]
    sc <- if (n$inactive) col.inactive else stage.col[n$col]
    fill <- if (n$inactive) NA else grDevices::adjustcolor(sc, alpha.f = 0.15)
    rect(n$col - bw / 2, n$row - bh / 2, n$col + bw / 2, n$row + bh / 2, col = fill,
         border = sc, lty = if (is.na(n$cond)) 1 else 2, lwd = 1.2)
    l <- boxes[[i]]
    y0 <- n$row - (length(l) - 1) * lh / 2
    for (k in seq_along(l)) {
      tc <- if (n$inactive) col.inactive else switch(l[[k]][[3]], main = col.text,
                                                     value = col.value, cond = sc)
      text(n$col, y0 + (k - 1) * lh, l[[k]][[1]], font = l[[k]][[2]], cex = cex, col = tc)
    }
  }
  if (legend) {
    legend(mean(c(0.5, n.col + 0.5)), n.row + 0.55, xjust = 0.5, horiz = TRUE, bty = "n",
           legend = c("data", "stored settings", "repeated", "conditional step",
                      if (any(nd$inactive)) "not used by this fit"),
           lty = c(1, 3, 2, NA, if (any(nd$inactive)) NA),
           pch = c(NA, NA, NA, 0, if (any(nd$inactive)) 0),
           col = c(rep(col.arrows, 3), col.text, if (any(nd$inactive)) col.inactive),
           text.col = col.text, cex = min(1, max(0.7, 0.85 * cex)))
  }
  invisible(x)
}

# Draw edges routed through the gaps between boxes so that no arrow crosses a
# box. Vertical runs use the channels between columns, horizontal runs the
# gaps between rows; edges sharing a channel are spread apart.
draw_map_edges <- function(edges, nd, bw, bh, col) {
  pos <- function(id) unlist(nd[nd$id == id, c("col", "row")])
  routes <- lapply(seq_len(nrow(edges)), function(i) {
    s <- pos(edges$from[i]); t <- pos(edges$to[i])
    list(s = s, t = t, type = edges$type[i])
  })

  # Channels used by each edge, to spread shared ones
  chan <- function(r) {
    s <- r$s; t <- r$t
    if (s[1] == t[1] && abs(s[2] - t[2]) == 1) return(character(0))
    if (s[1] == t[1]) return(paste0("v", s[1] + 0.5))
    if (t[1] - s[1] == 1) return(if (s[2] == t[2]) character(0) else paste0("v", s[1] + 0.5))
    c(paste0("v", s[1] + 0.5), paste0("h", t[2] - 0.5), paste0("v", t[1] - 0.5))
  }
  used <- lapply(routes, chan)
  offset <- list()
  for (ch in unique(unlist(used))) {
    who <- which(vapply(used, function(u) ch %in% u, logical(1)))
    offset[[ch]] <- setNames((seq_along(who) - (length(who) + 1) / 2) * 0.05, who)
  }
  off <- function(ch, i) offset[[ch]][[as.character(i)]]

  for (i in seq_along(routes)) {
    r <- routes[[i]]; s <- r$s; t <- r$t
    lty <- switch(r$type, data = 1, spec = 3, loop = 2)
    if (s[1] == t[1] && abs(s[2] - t[2]) == 1) {
      # Consecutive boxes in a column; a loop runs beside the forward arrow
      dx <- if (r$type == "loop") 0.12 else 0
      dir <- sign(t[2] - s[2])
      px <- c(s[1] + dx, s[1] + dx)
      py <- c(s[2] + dir * bh / 2, t[2] - dir * bh / 2)
    } else if (s[1] == t[1]) {
      cx <- s[1] + 0.5 + off(paste0("v", s[1] + 0.5), i)
      px <- c(s[1] + bw / 2, cx, cx, t[1] + bw / 2)
      py <- c(s[2], s[2], t[2], t[2])
    } else if (t[1] - s[1] == 1 && s[2] == t[2]) {
      px <- c(s[1] + bw / 2, t[1] - bw / 2)
      py <- c(s[2], t[2])
    } else if (t[1] - s[1] == 1) {
      cx <- s[1] + 0.5 + off(paste0("v", s[1] + 0.5), i)
      px <- c(s[1] + bw / 2, cx, cx, t[1] - bw / 2)
      py <- c(s[2], s[2], t[2], t[2])
    } else {
      c1 <- s[1] + 0.5 + off(paste0("v", s[1] + 0.5), i)
      gy <- t[2] - 0.5 + off(paste0("h", t[2] - 0.5), i)
      c2 <- t[1] - 0.5 + off(paste0("v", t[1] - 0.5), i)
      px <- c(s[1] + bw / 2, c1, c1, c2, c2, t[1] - bw / 2)
      py <- c(s[2], s[2], gy, gy, t[2], t[2])
    }
    k <- length(px)
    lines(px[-k], py[-k], col = col, lty = lty)
    arrows(px[k - 1], py[k - 1], px[k], py[k], length = 0.06, col = col, lty = lty)
  }
}
