rm(list = ls())

library(depmixS4)
library(ggplot2)

# ================== Step 1: Load and Prepare Your Data ==================
file_path <- "/Users/coenpotgieter/Documents/Hobbies/GitHub/skripsie/data/real/r_inp.csv"

# Read your CSV file
data <- read.csv(file_path)

# If your data has row names/numbers, you might need to reset the index
# data <- read.csv("your_file.csv", row.names = 1)

# Check the structure
str(data)
head(data)

# Ensure the data is numeric
data <- as.data.frame(apply(data, 2, as.numeric))


# ================== Step 2: Define HMMs with Different Numbers of States ==================
# Define models with different numbers of states
n_states <- 2:10  # Test from 2 to 6 hidden states

models <- list()

for (n in n_states) {
  cat("Fitting model with", n, "states...\n")

  # Define the HMM - all features are continuous so we use gaussian()
  models[[paste0("hmm_", n, "states")]] <- depmix(
    response = list(
      A1 ~ 1,
      A2 ~ 1,
      A3 ~ 1
    ),
    data = data,
    nstates = n,
    family = list(
      gaussian(),  # for A1
      gaussian(),  # for A2
      gaussian()   # for A3
    )
  )
}

# ================== Step 3: Fit All Models and Compare ==================
# Fit all models
fitted_models <- list()

for (i in 1:length(models)) {
  model_name <- names(models)[i]
  cat("Fitting", model_name, "...\n")
  
  # Fit with multiple random restarts to avoid local optima
  best_fit <- NULL
  best_logLik <- -Inf
  
  for (restart in 1:10) {
    tryCatch({
      fit_temp <- fit(models[[i]], verbose = FALSE)
      
      if (logLik(fit_temp) > best_logLik) {
        best_logLik <- logLik(fit_temp)
        best_fit <- fit_temp
      }
    }, error = function(e) {
      cat("Restart", restart, "failed for", model_name, "\n")
    })
  }
  
  if (!is.null(best_fit)) {
    fitted_models[[model_name]] <- best_fit
  }
}

# ================== Step 4: Model Selection ==================
# Create comparison table
results <- data.frame(
  m = sapply(fitted_models, function(x) x@nstates),
  log_lik = -2000 + sapply(fitted_models, logLik),
  aic = sapply(fitted_models, AIC),
  bic = sapply(fitted_models, BIC)
)

print(results)

write.csv(results, "/Users/coenpotgieter/Documents/Hobbies/GitHub/skripsie/data/real/r_model_selection.csv")

# Find the best model according to BIC (usually preferred for HMMs)
best_model_idx <- which.min(results$bic)
best_model <- fitted_models[[best_model_idx]]
n_best_states <- results$m[best_model_idx]
cat("\nBest model:", results$Model[best_model_idx], 
    "with", results$m[best_model_idx], "states\n")



# ================== Step 5: Extract Viterbi Path ==================
# Get the Viterbi path (most likely state sequence)
viterbi_states <- posterior(best_model)$state

# Calculate log probability for first observation, NaN for rest
viterbi_output <- data.frame(
  St = viterbi_states,
  log_prob = -2000 + c(logLik(best_model), rep(NaN, length(viterbi_states) - 1))
)

# Write Viterbi output
write.csv(viterbi_output, 
          "/Users/coenpotgieter/Documents/Hobbies/GitHub/skripsie/data/real/r_viterbi_output.csv",
          row.names = TRUE)

cat("\nViterbi output saved.\n")
print(head(viterbi_output))
print(tail(viterbi_output))

# ================== Step 6: Extract MPM Rule (Posterior Probabilities) ==================
# Get posterior probabilities for each state
posterior_probs <- posterior(best_model)

# Extract only the state probability columns
# The posterior() function returns a dataframe with state probabilities in columns S1, S2, etc.
state_cols <- grep("^S[0-9]+$", names(posterior_probs), value = TRUE)
mpm_output <- posterior_probs[, state_cols]

# Rename columns to match your format (S_1, S_2, etc.)
colnames(mpm_output) <- paste0("S_", 1:n_best_states)

# Write MPM output
write.csv(mpm_output,
          "/Users/coenpotgieter/Documents/Hobbies/GitHub/skripsie/data/real/r_mpm_output.csv",
          row.names = TRUE)

cat("\nMPM output saved.\n")
print(head(mpm_output))
print(tail(mpm_output))

# ================== Summary ==================
cat("\n=== Summary ===\n")
cat("Best model has", n_best_states, "states\n")
cat("Log-likelihood:", logLik(best_model), "\n")
cat("AIC:", AIC(best_model), "\n")
cat("BIC:", BIC(best_model), "\n")
cat("\nFiles saved:\n")
cat("1. Model selection results: r_model_selection.csv\n")
cat("2. Viterbi path: r_viterbi_output.csv\n")
cat("3. MPM posterior probabilities: r_mpm_output.csv\n")


