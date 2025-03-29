@echo off
REM Quick Feature-Grouped VAE Experiment
REM This batch file runs a quick version of the Feature-Grouped VAE experiment

echo Running Quick Feature-Grouped VAE Experiment...

REM Check if results directory exists, create if not
if not exist "results\feature_grouped_vae" mkdir "results\feature_grouped_vae"

REM Run the experiment in quick mode
python meaning_transform/experiments/run_feature_grouped_vae.py ^
  --quick ^
  --epochs 5 ^
  --batch-size 32 ^
  --latent-dim 32 ^
  --base-compression 1.0 ^
  --learning-rate 0.001 ^
  --output-dir results/feature_grouped_vae/quick ^
  %*

echo Quick experiment completed!
echo Results saved to results/feature_grouped_vae/quick 