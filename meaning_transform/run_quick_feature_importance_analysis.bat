@echo off
REM Quick Feature Importance Analysis Experiment (reduced parameters for faster testing)

echo Running quick feature importance analysis experiment...

REM Set Python path - update this if needed
set PYTHON=python

REM Create results directory if it doesn't exist
if not exist "results\feature_importance" mkdir "results\feature_importance"

REM Run a smaller feature importance analysis for quicker testing
%PYTHON% meaning_transform/run_feature_importance_analysis.py ^
    --output-dir "results/feature_importance/quick_test" ^
    --epochs 10 ^
    --batch-size 64 ^
    --num-states 1000 ^
    --db-path "simulation.db" ^
    --latent-dims "64" ^
    --compression-levels "1.0" ^
    --semantic-weights "1.0" ^
    --permutation-iterations 15 ^
    --feature-groups "spatial,resource,status,performance,role" ^
    --gpu

echo Quick feature importance analysis completed! 