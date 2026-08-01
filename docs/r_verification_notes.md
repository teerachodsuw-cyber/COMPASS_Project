# R verification notes

R reference fixtures (Gray's test, Wolbers' C-index, Schoop's competing-risks Brier score)
were generated via a one-off Docker (`rocker/r-ver`) install of `cmprsk`/`pec`. This
environment is **not persistent** — it must be rebuilt (~15-20 min, no build caching between
runs) each time new fixtures are needed. A persistent, checked-in Dockerfile was attempted
but not completed; accepted as a known limitation given the fixtures themselves are verified
and committed. Future sessions needing new R-verified fixtures should expect to redo this
install.
