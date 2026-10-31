args <- commandArgs(TRUE)
a <- readRDS(args[1]); b <- readRDS(args[2])
for (nm in names(a)) {
  x <- a[[nm]]; y <- b[[nm]]
  if (is.character(x) || is.character(y)) {
    cat(sprintf("%-15s ORIG=%s NEW=%s\n", nm, if (is.character(x)) x else "ok", if (is.character(y)) y else "ok"))
    next
  }
  diffs <- character(0)
  for (f in union(names(x), names(y))) {
    if (!identical(x[[f]], y[[f]])) {
      ae <- isTRUE(all.equal(x[[f]], y[[f]], check.attributes = FALSE, tolerance = 0))
      num <- if (is.numeric(x[[f]]) && is.numeric(y[[f]]) && length(x[[f]]) == length(y[[f]]))
        sprintf("maxabs=%.3g", max(abs(unlist(x[[f]]) - unlist(y[[f]])))) else ""
      diffs <- c(diffs, sprintf("%s%s%s", f, if (ae) "(attr only)" else "", if (nzchar(num)) paste0("[", num, "]") else ""))
    }
  }
  cat(sprintf("%-15s %s\n", nm, if (length(diffs)) paste(diffs, collapse = ", ") else "IDENTICAL"))
}
