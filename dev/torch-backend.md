# Plan: torch backend for mlp()

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
