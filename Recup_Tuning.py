from ray.tune import ExperimentAnalysis

directory = "C:/Users/a7xlm/ray_results/train_model_ray_2025-02-22_16-45-16"  # Directory of the experiment
analysis = ExperimentAnalysis("directory")  # Load experiment data

# Get the best trial based on a metric (e.g., lowest loss)
best_trial = analysis.get_best_trial(metric="loss", mode="min")  
best_config = best_trial.config  # Best hyperparameters

print("Best Hyperparameters:", best_config)