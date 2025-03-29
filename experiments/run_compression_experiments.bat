@echo off
REM Compression Experiments for Meaning-Preserving Transformations

echo Running compression experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\compression_experiments" mkdir "results\compression_experiments"

REM Run compression experiments with appropriate parameters
%PYTHON% meaning_transform/run_compression_experiments.py ^
    --output-dir "results/compression_experiments" ^
    --epochs 30 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "simulation.db" ^
    --latent-dim 32 ^
    --gpu

echo Compression experiments completed! 