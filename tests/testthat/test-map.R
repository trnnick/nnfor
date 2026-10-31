all_maps <- function() {
  list(nnfor_map("mlp"), nnfor_map("elm"),
       nnfor_map("mlp", stage = "forecast"), nnfor_map("elm", stage = "forecast"))
}

test_that("the map reflects the code: every caller calls its function", {
  ns <- asNamespace("nnfor")
  for (map in all_maps()) {
    nd <- map$nodes
    for (i in which(!is.na(nd$ref))) {
      expect_true(exists(nd$ref[i], envir = ns, inherits = FALSE),
                  info = paste("function", nd$ref[i], "does not exist"))
      if (!is.na(nd$caller[i])) {
        calls <- all.names(body(get(nd$caller[i], envir = ns)))
        expect_true(nd$ref[i] %in% calls,
                    info = paste(nd$caller[i], "does not call", nd$ref[i], "- update R/map.R"))
      }
    }
  }
})

test_that("maps are well formed", {
  for (map in all_maps()) {
    expect_true(all(c(map$edges$from, map$edges$to) %in% map$nodes$id))
    expect_false(anyDuplicated(paste(map$nodes$col, map$nodes$row)) > 0)
    expect_false(anyDuplicated(map$nodes$id) > 0)
    expect_equal(length(map$stages), max(map$nodes$col))
  }
})

test_that("maps are annotated from a fitted network", {
  set.seed(1)
  fit <- quiet(elm(AirPassengers, reps = 2, barebone = TRUE))
  map <- nnfor_map(fit)
  nd <- map$nodes
  expect_equal(map$model, "elm")
  expect_true(all(nd$inactive[nd$id %in% c("random", "hiddenact", "xreg")]))
  expect_false(nd$inactive[nd$id == "fast"])
  expect_match(nd$value[nd$id == "periods"], "ff = 12")
  expect_output(print(nnfor_map(fit, stage = "forecast")), "not used")
})

test_that("plotting a map leaves user settings unchanged", {
  pdf(NULL); on.exit(dev.off())
  par(mfrow = c(1, 2), fg = "blue")
  before <- user_settings()
  for (map in all_maps()) plot(map, main = "", legend = FALSE)
  expect_identical(user_settings(), before)
})
