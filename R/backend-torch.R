# torch backend for mlp().
#
# Standard networks: the weights of all ensemble members are held in one
# tensor P (members x weights), laid out per member exactly as neuralnet lays
# out its weight vector (layer by layer, each layer an (inputs + 1) x outputs
# matrix in column order with the bias in the first row). The forward pass
# is a batched matrix product over members and the loss is the sum of the
# members' losses, so each member's gradient depends on its own error only.
# Optimizers that update weights element by element (Rprop, Adam, SGD,
# RMSprop, Adagrad) therefore train every member exactly as if it were
# trained on its own, at the cost of a single network.
#
# Custom architectures (hd given as a function): each member is its own
# module; all members share one optimizer and one backward pass.
#
# Reproducibility: starting weights of standard networks are drawn with R's
# rnorm() as neuralnet draws them, and torch's own generator (used by custom
# modules for initialisation and e.g. dropout) is seeded from R's, so
# set.seed() controls the results of both.

backend_torch <- function() {
  check_torch()
  list(
    name = "torch",
    train = torch_train,
    n_members = function(net) net$reps,
    predict = torch_predict,
    layers = function(net, r) if (net$type == "standard") net$sizes else NULL,
    input_names = function(net) net$input.names
  )
}

check_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("backend = \"torch\" needs the torch package: install.packages(\"torch\").", call. = FALSE)
  }
  if (!torch::torch_is_installed()) {
    stop("The torch libraries are not installed: run torch::install_torch().", call. = FALSE)
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Settings: neuralnet names are translated to their torch equivalents
# ---------------------------------------------------------------------------

torch_activations <- c(logistic = "sigmoid", sigmoid = "sigmoid", tanh = "tanh",
                       relu = "relu", leaky_relu = "leaky_relu", elu = "elu", selu = "selu",
                       gelu = "gelu", silu = "silu", swish = "silu", mish = "mish",
                       softplus = "softplus")

# Activation function on tensors. `act` is a name (neuralnet or torch) or a
# function of a tensor.
torch_activation <- function(act) {
  if (is.function(act)) return(act)
  if (!is.character(act) || length(act) != 1 || !act %in% names(torch_activations)) {
    stop("act.fct must be a function of a tensor or one of: ",
         paste(names(torch_activations), collapse = ", "), ".", call. = FALSE)
  }
  switch(torch_activations[[act]],
         sigmoid = torch::torch_sigmoid,
         tanh = torch::torch_tanh,
         relu = torch::nnf_relu,
         leaky_relu = torch::nnf_leaky_relu,
         elu = torch::nnf_elu,
         selu = torch::nnf_selu,
         gelu = torch::nnf_gelu,
         silu = function(x) x * torch::torch_sigmoid(x),
         mish = function(x) x * torch::torch_tanh(torch::nnf_softplus(x)),
         softplus = torch::nnf_softplus)
}

torch_algorithms <- c("rprop+" = "rprop", "rprop-" = "rprop", rprop = "rprop",
                      backprop = "sgd", sgd = "sgd", adam = "adam", adamw = "adamw",
                      rmsprop = "rmsprop", adagrad = "adagrad")

resolve_torch_algorithm <- function(algorithm) {
  if (identical(algorithm, "lbfgs")) {
    stop("algorithm = \"lbfgs\" is not supported: it couples the ensemble members, which are trained together.",
         call. = FALSE)
  }
  if (!is.character(algorithm) || length(algorithm) != 1 || !algorithm %in% names(torch_algorithms)) {
    stop("algorithm must be one of: ", paste(names(torch_algorithms), collapse = ", "), ".",
         call. = FALSE)
  }
  torch_algorithms[[algorithm]]
}

# Rprop settings, following neuralnet: initial step 0.1, step factors 0.5 and
# 1.2, steps limited to [1e-10, 0.1].
rprop_settings <- function(learningrate, learningrate.limit, learningrate.factor) {
  lim <- unlist(learningrate.limit %||% list(min = 1e-10, max = 0.1))
  fac <- unlist(learningrate.factor %||% list(minus = 0.5, plus = 1.2))
  list(lr = learningrate %||% 0.1, min = lim[[1]], max = lim[[2]], minus = fac[[1]], plus = fac[[2]])
}

# Optimizer for a list of parameters. The C++ ("ignite") implementations are
# used where torch provides them: they are much faster for small networks.
torch_optimizer <- function(algorithm, params, learningrate, rp) {
  switch(algorithm,
         rprop = torch::optim_rprop(params, lr = rp$lr, etas = c(rp$minus, rp$plus),
                                    step_sizes = c(rp$min, rp$max)),
         sgd = torch::optim_ignite_sgd(params, lr = learningrate %||% 0.01),
         adam = torch::optim_ignite_adam(params, lr = learningrate %||% 0.001),
         adamw = torch::optim_ignite_adamw(params, lr = learningrate %||% 0.001),
         rmsprop = torch::optim_ignite_rmsprop(params, lr = learningrate %||% 0.01),
         adagrad = torch::optim_ignite_adagrad(params, lr = learningrate %||% 0.01))
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

torch_train <- function(Y, X, hd, reps, act.fct = "logistic", algorithm = "rprop",
                        threshold = 0.01, stepmax = 1e5, learningrate = NULL,
                        learningrate.limit = NULL, learningrate.factor = NULL, ...) {
  ignored <- names(list(...))
  if (length(ignored)) {
    warning("Arguments not used by the torch backend: ", paste(ignored, collapse = ", "), ".",
            call. = FALSE)
  }
  algorithm <- resolve_torch_algorithm(algorithm)
  act <- torch_activation(act.fct)
  # Seed torch's generator from R's so that set.seed() controls torch too.
  torch::torch_manual_seed(sample.int(.Machine$integer.max, 1))

  x <- torch::torch_tensor(X, dtype = torch::torch_float())
  y <- torch::torch_tensor(Y, dtype = torch::torch_float())
  settings <- list(act.fct = act.fct, algorithm = algorithm, threshold = threshold,
                   stepmax = stepmax, learningrate = learningrate)

  rp <- rprop_settings(learningrate, learningrate.limit, learningrate.factor)
  if (is.function(hd)) {
    net <- torch_train_custom(x, y, hd, reps, act, settings, rp)
  } else {
    net <- torch_train_standard(x, y, hd, reps, act, settings, rp)
  }
  net$input.names <- colnames(X)
  net$settings <- settings
  net$reps <- reps
  net
}

# Train until every member's largest absolute gradient is below threshold,
# or stepmax steps. `members_grad_max(grads)` returns that maximum per member
# and `freeze()` puts back the weights of members that have converged, so they
# no longer move (this also holds for optimizers with momentum).
#
# Gradients are computed functionally with autograd_grad() and handed to the
# optimizer explicitly, rather than read back from the parameters after
# zero_grad(): reading them back gave intermittently zero gradients with the
# C++ optimizers.
torch_fit_loop <- function(loss_fun, params, members_grad_max, freeze, opt, reps, settings) {
  converged <- torch::torch_zeros(reps, dtype = torch::torch_bool())
  steps <- torch::torch_zeros(reps, dtype = torch::torch_long())
  for (step in seq_len(settings$stepmax)) {
    grads <- torch::autograd_grad(loss_fun(), params)
    converged <- converged$logical_or(members_grad_max(grads) < settings$threshold)
    steps <- steps + converged$logical_not()
    # Synchronising with R is slow, so check for completion periodically.
    if (step %% 25 == 0 && as.logical(converged$all()$item())) break
    old <- lapply(params, function(p) p$detach()$clone())
    for (k in seq_along(params)) params[[k]]$grad <- grads[[k]]
    opt$step()
    torch::with_no_grad(freeze(old, converged))
  }
  list(converged = as.logical(torch::as_array(converged)),
       steps = as.integer(torch::as_array(steps)))
}

# `start` (members x weights) is only used by the tests.
#
# Speed: every torch call from R has a fixed cost that dominates for small
# networks, so the forward pass and loss are compiled once with jit_trace().
# Rprop, with the convergence check and freezing of converged members, is a
# single traced update; other algorithms use torch's C++ optimizers.
torch_train_standard <- function(x, y, hd, reps, act, settings, rp, start = NULL) {
  sizes <- c(ncol(x), hd, 1)
  n.w <- neuralnet_n_weights(ncol(x), hd)
  # Starting weights: drawn as neuralnet draws them, one member after another.
  sw <- start %||% matrix(stats::rnorm(n.w * reps), nrow = reps, byrow = TRUE)
  P <- torch::torch_tensor(sw, dtype = torch::torch_float(), requires_grad = TRUE)

  member_loss <- torch::jit_trace(function(P, x, y) {
    0.5 * (torch_forward_standard(P, x, sizes, act) - y)$pow(2)$sum(dim = c(2, 3))
  }, P, x, y)

  if (settings$algorithm == "rprop") {
    fit <- torch_rprop_standard(P, function(P) member_loss(P, x, y), reps, settings, rp)
    P <- fit$P
  } else {
    opt <- torch_optimizer(settings$algorithm, list(P), settings$learningrate, rp)
    fit <- torch_fit_loop(
      loss_fun = function() member_loss(P, x, y)$sum(),
      params = list(P),
      members_grad_max = function(grads) grads[[1]]$abs()$amax(dim = 2),
      freeze = function(old, converged) {
        P$copy_(torch::torch_where(converged$unsqueeze(2), old[[1]], P))
      },
      opt = opt, reps = reps, settings = settings)
  }

  error <- torch::with_no_grad(as.numeric(torch::as_array(member_loss(P, x, y))))
  list(type = "standard", sizes = sizes, P = torch::as_array(P$detach()),
       converged = fit$converged, steps = fit$steps, error = error, cache = new.env())
}

# Rprop for standard networks (torch's Rprop rule: a step grows when the
# gradient keeps its sign, shrinks and skips the update when it flips). A
# member whose largest absolute gradient is below threshold stops moving.
torch_rprop_standard <- function(P, member_loss, reps, settings, rp) {
  update <- torch::jit_trace(function(P, g, prev, step, conv, steps) {
    conv <- conv$logical_or(g$abs()$amax(dim = 2) < settings$threshold)
    g <- torch::torch_where(conv$unsqueeze(2), torch::torch_zeros_like(g), g)
    s <- (g * prev)$sign()
    step <- (step * torch::torch_where(s > 0, rp$plus, torch::torch_where(s < 0, rp$minus, 1)))$
      clamp(rp$min, rp$max)
    g <- torch::torch_where(s < 0, torch::torch_zeros_like(g), g)
    list(P - g$sign() * step, g, step, conv, steps + conv$logical_not())
  }, P$detach(), torch::torch_zeros_like(P), torch::torch_zeros_like(P),
  torch::torch_full_like(P, rp$lr), torch::torch_zeros(reps, dtype = torch::torch_bool()),
  torch::torch_zeros(reps, dtype = torch::torch_long()), strict = FALSE)

  prev <- torch::torch_zeros_like(P)
  step.size <- torch::torch_full_like(P, rp$lr)
  conv <- torch::torch_zeros(reps, dtype = torch::torch_bool())
  steps <- torch::torch_zeros(reps, dtype = torch::torch_long())
  for (i in seq_len(settings$stepmax)) {
    g <- torch::autograd_grad(member_loss(P)$sum(), P)[[1]]
    res <- update(P$detach(), g, prev, step.size, conv, steps)
    conv <- res[[4]]
    steps <- res[[5]]
    # Synchronising with R is slow, so check for completion periodically.
    if (i %% 25 == 0 && as.logical(conv$all()$item())) break
    P <- res[[1]]$requires_grad_(TRUE)
    prev <- res[[2]]
    step.size <- res[[3]]
  }
  list(P = P, converged = as.logical(torch::as_array(conv)),
       steps = as.integer(torch::as_array(steps)))
}

# Forward pass of standard networks whose weights are the rows of P.
# Returns members x n x 1.
torch_forward_standard <- function(P, x, sizes, act) {
  m <- P$size(1)
  h <- x$unsqueeze(1)$expand(c(m, -1, -1))
  off <- 0
  n.layers <- length(sizes) - 1
  for (l in seq_len(n.layers)) {
    n.in <- sizes[l] + 1
    n.out <- sizes[l + 1]
    # neuralnet layout: (inputs + 1) x outputs in column order, bias in row 1
    Wl <- P[, (off + 1):(off + n.in * n.out)]$view(c(m, n.out, n.in))$transpose(2, 3)
    h <- torch::torch_baddbmm(Wl[, 1:1, ], h, Wl[, 2:n.in, ])
    if (l < n.layers) h <- act(h)
    off <- off + n.in * n.out
  }
  h
}

# Custom architectures: `builder(n_inputs)` returns the hidden block as an
# nn_module; nnfor adds a linear output layer sized from a test pass.
torch_member_module <- function(builder, n_inputs) {
  block <- builder(n_inputs)
  width <- torch::with_no_grad(block(torch::torch_zeros(2, n_inputs))$size(2))
  torch::nn_sequential(block, torch::nn_linear(width, 1))
}

torch_train_custom <- function(x, y, builder, reps, act, settings, rp) {
  members <- lapply(seq_len(reps), function(r) torch_member_module(builder, ncol(x)))
  for (m in members) m$train()
  params <- lapply(members, function(m) m$parameters)
  member_loss <- function(r) 0.5 * (members[[r]](x) - y)$pow(2)$sum()
  opt <- torch_optimizer(settings$algorithm, do.call(c, params), settings$learningrate, rp)
  fit <- torch_fit_loop(
    loss_fun = function() torch::torch_stack(lapply(seq_len(reps), member_loss))$sum(),
    params = do.call(c, params),
    members_grad_max = function(grads) {
      k <- cumsum(c(0, lengths(params)))
      torch::torch_stack(lapply(seq_len(reps), function(r) {
        torch::torch_stack(lapply(grads[(k[r] + 1):k[r + 1]], function(g) g$abs()$max()))$max()
      }))
    },
    freeze = function(old, converged) {
      k <- 0
      for (r in seq_len(reps)) for (p in params[[r]]) {
        k <- k + 1
        p$copy_(torch::torch_where(converged[r], old[[k]], p))
      }
    },
    opt = opt, reps = reps, settings = settings)

  for (m in members) m$eval()
  error <- torch::with_no_grad(vapply(seq_len(reps), function(r) member_loss(r)$item(), numeric(1)))
  cache <- new.env()
  cache$live <- members
  list(type = "custom", builder = builder, n.inputs = ncol(x),
       state = lapply(members, function(m) torch_state_raw(m)),
       converged = fit$converged, steps = fit$steps, error = error, cache = cache)
}

torch_state_raw <- function(module) {
  con <- rawConnection(raw(0), "wb")
  on.exit(close(con))
  torch::torch_save(module$state_dict(), con)
  rawConnectionValue(con)
}

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

# Live torch objects are cached in net$cache. After a fit is saved and
# reloaded the cached objects are no longer valid; they are then rebuilt once
# from the stored weights.
torch_predict <- function(net, X, r) {
  x <- torch::torch_tensor(X, dtype = torch::torch_float())
  run <- function() {
    torch::with_no_grad({
      out <- if (net$type == "standard") {
        torch_forward_standard(torch_cached(net)[r:r, ], x, net$sizes,
                               torch_activation(net$settings$act.fct))
      } else {
        torch_cached(net)[[r]](x)
      }
    })
    as.numeric(torch::as_array(out))
  }
  tryCatch(run(), error = function(e) {
    if (is.null(net$cache$live)) stop(e)
    net$cache$live <- NULL
    run()
  })
}

torch_cached <- function(net) {
  if (is.null(net$cache$live)) {
    net$cache$live <- if (net$type == "standard") {
      torch::torch_tensor(net$P, dtype = torch::torch_float())
    } else {
      lapply(net$state, function(s) {
        m <- torch_member_module(net$builder, net$n.inputs)
        m$load_state_dict(torch::torch_load(s))
        m$eval()
        m
      })
    }
  }
  net$cache$live
}
