# Plan: new backends for mlp()

## What is already in place

All library-specific code in the MLP path goes through the backend
interface in `R/backend.R`. A backend is a list of functions:

| element             | purpose                                                       |
|---------------------|---------------------------------------------------------------|
| `name`              | stored in `fit$backend`                                       |
| `train(Y, X, hd, reps, ...)` | train all ensemble members, return one backend object |
| `n_members(net)`    | number of trained members                                     |
| `predict(net, X, r)`| numeric predictions of member `r` for the rows of matrix `X`  |
| `layers(net, r)`    | `c(inputs, hidden..., outputs)`, used by `plot_network()`     |
| `input_names(net)`  | input names in column order                                   |

`mlp()`, the hidden-node search (`select_hidden_cv`), `forecast()`,
`print()` and `plot()` only call this interface. Preprocessing is
backend-agnostic: it delivers scaled matrices `Y` (n x 1) and `X` (n x p,
named columns). ELM does not need a backend: it only uses neuralnet to draw
random hidden weights, which is unrelated to training.

`tests/testthat/test-backend.R` encodes the contract; a torch backend should
pass the same test.

## Implemented backends

Two backends implement this; they are worked examples for adding
another one.

**neuralnet** (`backend_neuralnet()` in `R/backend.R`). Members are trained
one by one, possibly in parallel through `nnfor_lapply()`, and merged into a
single neuralnet object.

**torch** (`backend_torch()` in `R/backend-torch.R`, torch in Suggests).

- Standard networks hold the weights of all members in one tensor (members x
  weights, in neuralnet's layout). The forward pass is a batched matrix
  product and the loss sums the members' losses, so each member's gradient
  depends on its own error only. Element-wise optimizers (Rprop, Adam, SGD,
  RMSprop, Adagrad) then train every member exactly as if alone; L-BFGS is
  excluded because it would couple them. A test checks this equivalence.
- Speed: each torch call from R has a fixed cost that dominates for small
  networks, so the forward pass and loss are compiled with `jit_trace()`, Rprop
  is a single traced update, and other optimizers use torch's C++ ("ignite")
  versions. Gradients are computed with `autograd_grad()` and passed to the
  optimizer explicitly.
- Custom architectures (`hd` given as a function returning an `nn_module`):
  one module per member, one shared optimizer and backward pass.
- Storage: weights are kept as R arrays (standard) or serialised state
  (custom); live torch objects are cached in an environment and rebuilt after
  the fit is reloaded.

## Rules for any backend

- **Reproducibility.** Draw all random numbers from R's generator before
  training (seed the library's own generator from it if needed), so that
  `set.seed()` controls results and they do not depend on `nnfor.cores`.
- **Settings in `...`.** Accept neuralnet's argument names where the meaning
  is the same, translate names internally, and warn about unused arguments.
- **Members.** `n_members()` must match the number of members `predict()`
  accepts; dropping or keeping non-converged members is the backend's choice
  but must be documented.
- **Test.** Add the backend to `tests/testthat/test-backend.R` or an
  equivalent file, skipped when the library is not installed.
