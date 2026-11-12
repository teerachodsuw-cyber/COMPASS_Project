# Fine-Gray subdistribution hazard model for one outcome, death competing:
# survival::finegray() expansion + weighted coxph() with a fixed ridge penalty.
# Called by compass.models.cox_baseline.fit_fine_gray (training split only). Verified against
# cmprsk::crr (tests/test_cox_baseline.py).
#
# Usage: Rscript fine_gray_fit.R <in.csv> <out.json> <theta> <ties> <grid: comma-separated years>
#   in.csv: time (years), status (0 censored / 1 event / 2 competing death), then one column per
#           standardised feature. Patient-level, written to a temporary directory by the caller.
#   theta:  ridge penalty on the summed log partial likelihood, theta / 2 * sum(beta^2); 0 = none.
#   out.json (aggregates only): coefficients, the baseline cumulative subdistribution hazard at
#           x = 0 evaluated on the grid, counts, warnings, package versions.
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; outfile <- args[2]; theta <- as.numeric(args[3]); ties <- args[4]
grid <- as.numeric(strsplit(args[5], ",")[[1]])

d <- read.csv(infile, check.names = FALSE)
xcols <- setdiff(names(d), c("time", "status"))
safe <- sprintf("x%d", seq_along(xcols))          # syntactic names for the formula
names(d)[match(xcols, names(d))] <- safe
d$status <- factor(d$status, levels = 0:2, labels = c("censor", "event", "death"))

warns <- character(0)
fit <- withCallingHandlers({
  fg <- finegray(Surv(time, status) ~ ., data = d, etype = "event")
  rhs <- if (theta > 0) {
    sprintf("ridge(%s, theta = %.17g, scale = FALSE)", paste(safe, collapse = ", "), theta)
  } else {
    paste(safe, collapse = " + ")
  }
  coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", rhs)), data = fg, weights = fgwt,
        ties = ties, robust = FALSE)
}, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })

beta <- unname(coef(fit))
bh <- basehaz(fit, centered = FALSE)                  # cumulative baseline subdistribution hazard at x = 0
h0 <- sapply(grid, function(t) { k <- bh$time <= t; if (any(k)) max(bh$hazard[k]) else 0 })

write_json(list(
  features = xcols,
  coefficients = beta,
  grid_years = grid,
  baseline_cumhaz = h0,
  theta = theta,
  ties = ties,
  n = nrow(d),
  n_events = sum(d$status == "event"),
  n_competing = sum(d$status == "death"),
  iterations = fit$iter,
  warnings = warns,
  r_version = R.version.string,
  survival_version = as.character(packageVersion("survival"))
), outfile, auto_unbox = TRUE, digits = NA)
