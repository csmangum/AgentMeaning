@echo off
REM Compression Experiments with Beta Annealing for Meaning-Preserving Transformations

echo Running compression experiments with beta annealing...

REM Set Python path - update this if needed
set PYTHON=python

REM Set Python module path to include project root
set PYTHONPATH=%CD%

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\beta_compression_experiments" mkdir "results\beta_compression_experiments"

REM Run compression experiments with appropriate parameters and beta annealing enabled
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/beta_compression_experiments" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "0.5,1.0,2.0,5.0" ^
    --beta-annealing ^
    --gpu

echo Beta annealing compression experiments completed!

REM Run adaptive model experiments with beta annealing
echo Running adaptive model compression experiments with beta annealing...
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/beta_compression_experiments/adaptive" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "0.5,1.0,2.0,5.0" ^
    --adaptive ^
    --beta-annealing ^
    --gpu

echo Adaptive beta annealing compression experiments completed! 