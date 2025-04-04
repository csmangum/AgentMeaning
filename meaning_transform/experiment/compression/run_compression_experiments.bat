@echo off
REM Compression Experiments for Meaning-Preserving Transformations

echo Running compression experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Set Python module path to include project root
set PYTHONPATH=%CD%

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\compression_experiments" mkdir "results\compression_experiments"

REM Run compression experiments with appropriate parameters
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/compression_experiments" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,5.0,10.0,20.0" ^
    --gpu

echo Standard compression experiments completed!

REM Run adaptive model experiments 
echo Running adaptive model compression experiments...
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/compression_experiments/adaptive" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "0.1,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,5.0,10.0,20.0" ^
    --adaptive ^
    --gpu

echo Adaptive compression experiments completed! 