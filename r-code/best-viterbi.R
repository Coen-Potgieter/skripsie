rm(list = ls())

library(depmixS4)
library(ggplot2)

# ================== Configuration ==================
file_path <- "/Users/coenpotgieter/Documents/Hobbies/GitHub/skripsie/data/real/r_inp.csv"
output_dir <- "/Users/coenpotgieter/Documents/Hobbies/GitHub/skripsie/data/real/models"
n_states <- 6
n_models <- 10000  # Number of independent models to create

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ================== Step 1: Load and Prepare Data ==================
data <- read.csv(file_path)

# Ensure the data is numeric
data <- as.data.frame(apply(data, 2, as.numeric))

cat("Data loaded successfully\n")
cat("Data dimensions:", nrow(data), "rows x", ncol(data), "columns\n\n")

# ================== Step 2: Fit Multiple Independent Models ==================
for (i in 1:n_models) {
  cat("========================================\n")
  cat("Fitting Model", i, "of", n_models, "...\n")
  cat("========================================\n")
  
  tryCatch({
    # Define the HMM
    hmm_model <- depmix(
      response = list(
        A1 ~ 1,
        A2 ~ 1,
        A3 ~ 1
      ),
      data = data,
      nstates = n_states,
      family = list(
        gaussian(),  # for A1
        gaussian(),  # for A2
        gaussian()   # for A3
      )
    )
    
    # Fit the model (single run, no restarts)
    fitted_model <- fit(hmm_model, verbose = FALSE)
    
    # Extract results
    log_lik <- logLik(fitted_model)
    aic_val <- AIC(fitted_model)
    bic_val <- BIC(fitted_model)
    
    cat("Log-likelihood:", log_lik, "\n")
    cat("AIC:", aic_val, "\n")
    cat("BIC:", bic_val, "\n")
    
    # ================== Save Model Statistics ==================
    model_stats <- data.frame(
      model_id = i,
      n_states = n_states,
      log_lik = log_lik,
      aic = aic_val,
      bic = bic_val
    )
    
    # write.csv(model_stats, file.path(output_dir, paste0("model_", i, "_stats.csv")), row.names = FALSE)
    
    # ================== Save Viterbi Path ==================
    viterbi_states <- posterior(fitted_model)$state
    
    viterbi_output <- data.frame(
      St = viterbi_states,
      log_prob = c(log_lik, rep(NaN, length(viterbi_states) - 1))
    )
    
    write.csv(viterbi_output, 
              file.path(output_dir, paste0("model_", i, "_viterbi.csv")),
              row.names = FALSE)
    
    # ================== Save MPM Posterior Probabilities ==================
    posterior_probs <- posterior(fitted_model)
    
    # Extract state probability columns
    state_cols <- grep("^S[0-9]+$", names(posterior_probs), value = TRUE)
    mpm_output <- posterior_probs[, state_cols]
    
    # Rename columns
    colnames(mpm_output) <- paste0("S_", 1:n_states)
    
    write.csv(mpm_output,
              file.path(output_dir, paste0("model_", i, "_mpm.csv")),
              row.names = FALSE)
    
    cat("Model", i, "completed successfully!\n\n")
    
  }, error = function(e) {
    cat("ERROR: Model", i, "failed with error:", e$message, "\n\n")
  })
}

# ================== Summary ==================
cat("\n========================================\n")
cat("=== All Models Complete ===\n")
cat("========================================\n")
cat("Total models fitted:", n_models, "\n")
cat("Number of states per model:", n_states, "\n")
cat("Output directory:", output_dir, "\n")
cat("\nFor each model, the following files were saved:\n")
cat("2. model_X_viterbi.csv - Viterbi path (most likely state sequence)\n")
cat("3. model_X_mpm.csv - MPM posterior probabilities\n")