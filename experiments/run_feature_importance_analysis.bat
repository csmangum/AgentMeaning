@echo off
REM Feature Importance Analysis Experiment for Meaning-Preserving Transformations

echo Running feature importance analysis experiment...

REM Set Python path - update this if needed
set PYTHON=python

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\feature_importance" mkdir "results\feature_importance"

REM Run feature importance analysis with appropriate parameters
%PYTHON% meaning_transform/run_feature_importance_analysis.py ^
    --output-dir "results/feature_importance" ^
    --epochs 30 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "simulation.db" ^
    --latent-dims "64" ^
    --compression-levels "1.0" ^
    --semantic-weights "1.0" ^
    --permutation-iterations 10 ^
    --feature-groups "spatial,resource,status,performance,role" ^
    --gpu

echo Feature importance analysis completed! 