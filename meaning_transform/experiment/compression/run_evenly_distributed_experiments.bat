@echo off
REM Evenly Distributed Compression Experiments for Meaning-Preserving Transformations

echo Running evenly distributed compression experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Set Python module path to include project root
set PYTHONPATH=%CD%

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\even_compression" mkdir "results\even_compression"

REM Run compression experiments with evenly distributed compression levels
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/even_compression" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0" ^
    --gpu

echo Standard evenly distributed compression experiments completed!

REM Run adaptive model experiments 
echo Running adaptive evenly distributed compression experiments...
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/even_compression/adaptive" ^
    --epochs 50 ^
    --batch-size 64 ^
    --num-states 5000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 32 ^
    --compression-levels "10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0" ^
    --gpu ^
    --adaptive

echo Adaptive evenly distributed compression experiments completed! 