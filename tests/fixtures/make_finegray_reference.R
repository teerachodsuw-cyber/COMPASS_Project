# Generates tests/fixtures/finegray_reference_data.csv and finegray_crr_reference.csv: a small
# simulated competing-risks data set (no tied times) and the Fine-Gray CIF predicted by
# cmprsk::crr (an implementation independent of survival::finegray + coxph) at yearly times for
# three covariate vectors. Run once from the repo root: Rscript tests/fixtures/make_finegray_reference.R
suppressMessages(library(cmprsk))
set.seed(20261112)
n <- 300
x1 <- round(rnorm(n), 6); x2 <- rbinom(n, 1, 0.4)
t1 <- rexp(n, 0.08 * exp(0.6 * x1 - 0.4 * x2)); t2 <- rexp(n, 0.05 * exp(0.3 * x2)); cz <- runif(n, 1, 12)
time <- round(pmin(t1, t2, cz), 8)
status <- ifelse(time == round(t1, 8), 1, ifelse(time == round(t2, 8), 2, 0))
stopifnot(!any(duplicated(time)))
write.csv(data.frame(time, status, x1, x2), "tests/fixtures/finegray_reference_data.csv", row.names = FALSE)
fit <- crr(time, status, cbind(x1, x2), failcode = 1, cencode = 0)
cov <- rbind(c(0, 0), c(1, 0), c(-0.5, 1))
pred <- predict(fit, cov1 = cov)
out <- do.call(rbind, lapply(1:5, function(t) {
  k <- max(which(pred[, 1] <= t))
  data.frame(t = t, x1 = cov[, 1], x2 = cov[, 2], cif = pred[k, -1])
}))
write.csv(out, "tests/fixtures/finegray_crr_reference.csv", row.names = FALSE)
cat("crr coefficients:", fit$coef, "\n")
