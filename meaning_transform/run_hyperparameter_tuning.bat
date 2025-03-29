@echo off
REM Hyperparameter Tuning Experiments for Meaning-Preserving Transformations

echo Running hyperparameter tuning experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Create results directory if it doesn't exist
if not exist "results\hyperparameter_tuning" mkdir "results\hyperparameter_tuning"

REM Run hyperparameter tuning with appropriate parameters
%PYTHON% meaning_transform/run_hyperparameter_tuning.py ^
    --output-dir "results/hyperparameter_tuning" ^
    --epochs 30 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "simulation.db" ^
    --latent-dims "16,32,64,128" ^
    --compression-levels "0.5,1.0,2.0" ^
    --semantic-weights "0.1,0.5,1.0,2.0" ^
    --gpu

echo Hyperparameter tuning completed! 