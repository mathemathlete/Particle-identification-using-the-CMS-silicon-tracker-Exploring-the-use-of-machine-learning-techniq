from ray.tune import ExperimentAnalysis

analysis = ExperimentAnalysis("C:/Users/Kamil/ray_results/train_model_ray_2025-02-14_10-21-50")  # Load experiment data

# Get the best trial based on a metric (e.g., lowest loss)
best_trial = analysis.get_best_trial(metric="loss", mode="min")  
best_config = best_trial.config  # Best hyperparameters

print("Best Hyperparameters:", best_config)