@echo off
REM Quick Hyperparameter Tuning Experiments (subset of full experiments)

echo Running quick hyperparameter tuning experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Create results directory if it doesn't exist
if not exist "results\hyperparameter_tuning" mkdir "results\hyperparameter_tuning"

REM Run a smaller hyperparameter search for quicker testing
%PYTHON% meaning_transform/run_hyperparameter_tuning.py ^
    --output-dir "results/hyperparameter_tuning/quick_test" ^
    --epochs 10 ^
    --batch-size 64 ^
    --num-states 1000 ^
    --db-path "simulation.db" ^
    --latent-dims "32,64" ^
    --compression-levels "0.5,1.0" ^
    --semantic-weights "0.5,1.0" ^
    --gpu

echo Quick hyperparameter tuning completed! 