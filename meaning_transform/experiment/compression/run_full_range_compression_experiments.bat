@echo off
REM Full-Range Compression Experiments for Meaning-Preserving Transformations

echo Running full-range compression experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Set Python module path to include project root
set PYTHONPATH=%CD%

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\full_range_compression" mkdir "results\full_range_compression"

REM Run compression experiments with appropriate parameters
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/full_range_compression" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "0.001,0.005,0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,7.5,10.0,15.0,20.0,30.0,50.0,75.0,100.0" ^
    --gpu

echo Standard full-range compression experiments completed!

REM Run adaptive model experiments 
echo Running adaptive full-range compression experiments...
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/full_range_compression/adaptive" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "0.001,0.005,0.01,0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,7.5,10.0,15.0,20.0,30.0,50.0,75.0,100.0" ^
    --gpu ^
    --adaptive

echo Adaptive full-range compression experiments completed! 