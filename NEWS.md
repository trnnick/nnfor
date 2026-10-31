# CHANGES IN NNFOR VERSION 0.9.9
- Major refactoring: Preprocessing split into focused steps (differencing and 
  scaling, lagged inputs, seasonal dummies, input selection). Training and 
  forecasting build inputs with the same function, so they cannot disagree.
- Network training goes through a backend interface (neuralnet for now) to
  prepare for further backends, such as torch.
- Plotting consolidated. Plots no longer override user graphical settings
  (`par()`, layouts, colours) and default line colours follow `par("fg")`.
- New:`nnfor_map()` shows how information flows through the package when fitting
  (`stage = "fit"`) or forecasting (`stage = "forecast"`) with `mlp()` or
  `elm()`. Plot it for a diagram or print it for an outline. Given a fitted
  network, the map shows the values that fit used and greys out unused steps.
- `m` when used, it overrides the seasonal periods of `y` (vector for
  multiple seasonalities). Its default is now `NULL`. The periods are stored 
  in the fitted object as `ff`.
- Plots accept `...` for the plot frame and `*.args` lists for each drawn
  component. Time series plots accept `add = TRUE`.
- `plot(fit, which = "fit")` reproduces the `outplot` figure after fitting.
  `plot_network()` and `plot_fit()` are exported.
- `plot(forecast)` can show the in-sample fit (`fitted = TRUE`). Default
  y-limits follow `xlim`.
- Fitted objects store the fit of each ensemble member (`fitted.all`).
- Ensemble members can be trained in parallel with `options(nnfor.cores = )`,
  giving an integer or a cluster from `parallel::makeCluster()`. This covers
  MLP training, the hidden-node search, ELM output weights and `elm.fast()`.
  Random numbers are drawn before members are dispatched, so results are
  identical for any number of cores and to sequential runs.
- Requires tsutils >= 0.9.5, whose seasdummy() fixes binary seasonal dummies
  that were out of phase for the last observations of the sample when it did
  not start at the first season. Fits with binary dummies change as a result.
- When input selection rejects every univariate lag, but regressors or
  dummies remain, the network uses those only; previously all lags were kept.
- `elm.fast()` logistic output applies the logistic function (the inverse
  link of the binomial model the output weights are estimated with) and
  reverses the stored target scaling, as linear output does. Previously the
  hidden-layer activation was used and predictions were rescaled by their
  own range (a single prediction gave NaN).
- Reusing a model (`model =`) copies its specification exactly, rather than
  re-detecting differencing and seasonality on the new data.
- `mlp()` failed with "argument is of length zero" when only some ensemble
  members converged (a neuralnet issue). Members that do not converge are
  now dropped, with a warning, as intended.
- Dummies rebuilt after input selection used the wrong seasonal periods for
  multiple seasonalities.
- `mlp.thief()` and `elm.thief()` passed `xreg` as `y` to `forecast()`.
- `elm(barebone = TRUE, type = "lm")` failed; `"lm"` and `"ls"` are now
  synonyms.
- ELM with several hidden layers used wrong weights.
- ELM network plot greyed out the wrong hidden nodes.
- One-step forecasts with trigonometric dummies failed.
- Clear errors when forecasting without required `xreg`, when no network
  converges, and for `barebone = TRUE` with several hidden layers.

# CHANGED IN NNFOR VERSION 0.9.7 (07 JUL 2022)
- Added warning when predict() is called for ELM and MLP. 

# CHANGED IN NNFOR VERSION 0.9.7 (05 JUL 2022)
- Updated dependencies

# CHANGES IN NNFOR VERSION 0.9.6 (15 JAN 2019)
- Updated dependencies

# CHANGES IN NNFOR VERSION 0.9.5 (12 JAN 2019)
- Removed functions mseastest, cmav and lagmatrix from nnfor. These are now available in package tsutils.
- Updated license to GPL-3.

# CHANGES IN NNFOR VERSION 0.9.4 (11 JUL 2018)
- Fix for msts time series and allow.det.season argument.

# CHANGES IN NNFOR VERSION 0.9.3 (01 Apr 2018)
- Changes in how trigonometric seasonal dummies are handled.

# CHANGES IN NNFOR VERSION 0.9.2 (11 Dec 2017)
- Documentation fixes.

# CHANGES IN NNFOR VERSION 0.9.2 (30 Nov 2017)
- Added option to use previously specified mlp and elm (see arguments model and retrain).

# CHANGES IN NNFOR VERSION 0.9.1 (29 Oct 2017)
- Fix on univariate keep vector when no univariate lags are selected (lags=0). 
