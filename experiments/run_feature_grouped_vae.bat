@echo off
REM Feature-Grouped VAE Experiment
REM This batch file runs the Feature-Grouped VAE experiment

echo Running Feature-Grouped VAE Experiment...

REM Check if results directory exists, create if not
if not exist "results\feature_grouped_vae" mkdir "results\feature_grouped_vae"

REM Run the experiment
python meaning_transform/experiments/run_feature_grouped_vae.py ^
  --epochs 50 ^
  --batch-size 64 ^
  --latent-dim 32 ^
  --base-compression 1.0 ^
  --learning-rate 0.001 ^
  --output-dir results/feature_grouped_vae ^
  %*

echo Experiment completed!
echo Results saved to results/feature_grouped_vae 