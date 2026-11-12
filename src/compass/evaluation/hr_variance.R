# Model-based variance of a STORED penalised Cox / Fine-Gray fit, evaluated at the stored
# coefficients (no re-estimation): coxph(..., init = beta, iter.max = 0) with the same ridge
# penalty, ties and data as the original fit. Called by compass.evaluation.importance (training
# split only; the patient-level input lives in a temporary directory).
#
# Usage: Rscript hr_variance.R <in.csv> <beta.json> <out.json> <theta> <ties> <cox|finegray>
#   in.csv: time (years), status (0 censored / 1 event / 2 competing death), then the
#           standardised feature columns in the fit's column order.
#   out.json (aggregates only): coefficients as returned (must equal the stored ones),
#           var (inverse penalised information: coxph's default), var2 (sandwich I^-1 J I^-1).
suppressMessages(library(survival))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; betafile <- args[2]; outfile <- args[3]; theta <- as.numeric(args[4]); ties <- args[5]
mode <- args[6]

d <- read.csv(infile, check.names = FALSE)
beta <- as.numeric(fromJSON(betafile))
xcols <- setdiff(names(d), c("time", "status"))
safe <- sprintf("x%d", seq_along(xcols))
names(d)[match(xcols, names(d))] <- safe
rhs <- sprintf("ridge(%s, theta = %.17g, scale = FALSE)", paste(safe, collapse = ", "), theta)
ctrl <- coxph.control(iter.max = 0)

warns <- character(0)
fit <- withCallingHandlers({
  if (mode == "cox") {
    d$event <- as.integer(d$status == 1)
    coxph(as.formula(paste("Surv(time, event) ~", rhs)), data = d, ties = ties, init = beta, control = ctrl)
  } else {
    d$status <- factor(d$status, levels = 0:2, labels = c("censor", "event", "death"))
    fg <- finegray(Surv(time, status) ~ ., data = d, etype = "event")
    coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", rhs)), data = fg, weights = fgwt, ties = ties,
          init = beta, control = ctrl, robust = FALSE)
  }
}, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })

write_json(list(
  coefficients = unname(coef(fit)),
  var = unname(fit$var),
  var2 = if (is.null(fit$var2)) NULL else unname(fit$var2),
  iterations = fit$iter,
  warnings = warns,
  survival_version = as.character(packageVersion("survival"))
), outfile, auto_unbox = TRUE, digits = NA)
