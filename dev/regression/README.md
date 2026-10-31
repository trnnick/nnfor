# Regression harness

Runs a fixed set of seeded fits and forecasts (MLP/ELM, xreg, hidden-node
search, dummies, msts, model reuse, input-selection fallbacks, ...) and saves
everything numeric a user could observe, so that two builds of nnfor can be
compared exactly.

    R CMD INSTALL -l /tmp/lib_a <old source>
    R CMD INSTALL -l /tmp/lib_b <new source>
    Rscript dev/regression/cases.R /tmp/lib_a a.rds
    Rscript dev/regression/cases.R /tmp/lib_b b.rds
    Rscript dev/regression/compare.R a.rds b.rds

Each case prints IDENTICAL, or the fields that differ with the largest
absolute difference. Results depend on the versions of neuralnet, glmnet and
forecast, so compare builds on the same machine.
