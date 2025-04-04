@echo off
REM Feature Importance Hierarchy Robustness Analysis
REM Implementation of Step 18 in the project roadmap (clean version without pip)

REM Create output directory
mkdir findings\feature_importance_robustness 2>nul

REM Run the analysis
python experiments\run_feature_importance_robustness.py ^
  --num_states 5000 ^
  --n_folds 5 ^
  --num_epochs 30 ^
  --batch_size 64 ^
  --learning_rate 0.001 ^
  --state_dim 64 ^
  --latent_dim 32 ^
  --random_seed 42 ^
  --output_dir findings\feature_importance_robustness ^
  --use_gpu

echo Feature Importance Hierarchy Robustness Analysis completed!
echo Results available in the findings\feature_importance_robustness directory 