# Training backends for MLP networks.
#
# Everything that depends on the library used to train the network goes
# through this interface. mlp(), the hidden-node search, forecasting, print
# and plot only ever call these functions, so adding a backend (e.g. torch)
# means writing one constructor below and registering it in nn_backend().
#
# A backend is a list with:
#   name                      character, stored in the fitted object
#   train(Y, X, hd, reps, ...) -> backend fit holding all ensemble members.
#                                 Y is an n x 1 matrix, X an n x p matrix with
#                                 column names; both are already scaled.
#                                 `...` are backend-specific training options.
#   n_members(net)             -> number of successfully trained members
#   predict(net, X, r)         -> numeric vector: predictions of member r for
#                                 the rows of X (X is always a matrix, possibly
#                                 with a single row during forecasting)
#   layers(net, r)             -> layer sizes of member r:
#                                 c(inputs, hidden_1, ..., hidden_k, outputs)
#   input_names(net)           -> character vector of input names, in order
#
# The training data are passed as plain matrices so that preprocessing stays
# backend-agnostic, and all members live inside one backend object so that
# the fitted mlp object has the same shape regardless of backend.

nn_backend <- function(name = "neuralnet") {
  switch(name,
         neuralnet = backend_neuralnet(),
         stop("Unknown network backend: ", name, call. = FALSE))
}

# Backend of a fitted object. Objects created before backends existed were
# all trained with neuralnet.
object_backend <- function(object) nn_backend(object$backend %||% "neuralnet")

network_formula <- function(X) {
  as.formula(paste0("Y~", paste0(colnames(X), collapse = "+")), env = baseenv())
}

backend_neuralnet <- function() {
  list(
    name = "neuralnet",

    # Members are trained one by one (in parallel if options(nnfor.cores)
    # allows) from starting weights drawn up front, exactly as neuralnet draws
    # them rep by rep, so results do not depend on the number of cores. This
    # also lets members that fail to converge be dropped: neuralnet 1.44.2
    # fails when only some of several reps converge.
    train = function(Y, X, hd, reps, ...) {
      frm <- network_formula(X)
      dat <- cbind(Y, X)
      if (any(c("startweights", "exclude", "constant.weights") %in% names(list(...)))) {
        # Initial weights controlled by the user: one call to neuralnet.
        net <- neuralnet::neuralnet(frm, dat, hidden = hd, rep = reps, err.fct = "sse",
                                    linear.output = TRUE, ...)
      } else {
        n.w <- neuralnet_n_weights(ncol(X), hd)
        sw <- stats::rnorm(n.w * reps)
        members <- nnfor_lapply(seq_len(reps), function(r) {
          withCallingHandlers(
            neuralnet::neuralnet(frm, dat, hidden = hd, rep = 1,
                                 startweights = sw[(r - 1) * n.w + seq_len(n.w)],
                                 err.fct = "sse", linear.output = TRUE, ...),
            warning = function(w) {
              if (grepl("did not converge", conditionMessage(w))) invokeRestart("muffleWarning")
            })
        })
        net <- merge_neuralnets(members, reps)
        failed <- reps - length(net$weights)
        if (failed > 0) {
          warning(sprintf("Algorithm did not converge in %s of %s repetition(s) within the stepmax.",
                          failed, reps), call. = FALSE)
        }
      }
      if (length(net$weights) == 0) {
        stop("None of the networks converged. Consider increasing `stepmax' or `threshold' (passed to neuralnet via ...).",
             call. = FALSE)
      }
      net
    },

    n_members = function(net) length(net$weights),

    predict = function(net, X, r) {
      as.vector(neuralnet::compute(net, X, r)$net.result)
    },

    layers = function(net, r) {
      w <- net$weights[[r]]
      c(nrow(w[[1]]) - 1, vapply(w, ncol, integer(1)))  # -1 for the bias row
    },

    input_names = function(net) net$model.list$variables
  )
}

# Number of weights (including biases) of a network with p inputs, hidden
# layers hd and one output; the number neuralnet draws per rep.
neuralnet_n_weights <- function(p, hd) {
  sum((c(p, hd) + 1) * c(hd, 1))
}

# Combine single-rep neuralnet fits into one object, as neuralnet(rep = reps)
# would return it. Members that did not converge are dropped.
merge_neuralnets <- function(members, reps) {
  ok <- vapply(members, function(m) length(m$weights) > 0, logical(1))
  net <- members[[1]]
  for (f in c("net.result", "weights", "generalized.weights", "startweights")) {
    net[f] <- list(do.call(c, lapply(members[ok], `[[`, f)))
  }
  net["result.matrix"] <- list(do.call(cbind, lapply(members[ok], `[[`, "result.matrix")))
  net$call$rep <- reps
  net$call$startweights <- NULL
  net
}
