@echo off
REM Quick Feature Importance Hierarchy Robustness Analysis
REM Implementation of Step 18 in the project roadmap (reduced parameters for speed)

REM Create output directory
mkdir findings\feature_importance_robustness_quick 2>nul

REM Run the analysis with reduced parameters for quick testing
python experiments\run_feature_importance_robustness.py ^
  --num_states 1000 ^
  --n_folds 3 ^
  --num_epochs 5 ^
  --batch_size 64 ^
  --learning_rate 0.001 ^
  --state_dim 32 ^
  --latent_dim 16 ^
  --random_seed 42 ^
  --output_dir findings\feature_importance_robustness_quick ^
  --use_gpu

echo Quick Feature Importance Hierarchy Robustness Analysis completed!
echo Results available in the findings\feature_importance_robustness_quick directory 