@echo off
REM Quick Compression Experiments for Meaning-Preserving Transformations (reduced parameters)

echo Running quick compression experiments...

REM Set Python path - update this if needed
set PYTHON=python

REM Set Python module path to include project root
set PYTHONPATH=%CD%

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\compression_experiments" mkdir "results\compression_experiments"

REM Run standard quick compression experiments
echo Running standard quick compression experiments...
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/compression_experiments/quick_test" ^
    --epochs 5 ^
    --batch-size 32 ^
    --num-states 1000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 16 ^
    --compression-levels "0.5,2.0,5.0" ^
    --skip-drift ^
    --gpu ^
    --debug

echo Standard quick compression experiments completed!

REM Run adaptive model quick compression experiments
echo Running adaptive quick compression experiments...
%PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
    --output-dir "results/compression_experiments/quick_test_adaptive" ^
    --epochs 5 ^
    --batch-size 32 ^
    --num-states 1000 ^
    --db-path "data/simulation.db" ^
    --latent-dim 16 ^
    --compression-levels "0.5,2.0,5.0" ^
    --adaptive ^
    --skip-drift ^
    --gpu ^
    --debug

echo Adaptive quick compression experiments completed!

REM Run graph model quick compression experiments (if needed)
REM echo Running graph-based quick compression experiments...
REM %PYTHON% meaning_transform/experiment/compression/run_compression_experiments.py ^
REM     --output-dir "results/compression_experiments/quick_test_graph" ^
REM     --epochs 5 ^
REM     --batch-size 16 ^
REM     --num-states 500 ^
REM     --db-path "data/simulation.db" ^
REM     --latent-dim 16 ^
REM     --compression-levels "0.5,2.0,5.0" ^
REM     --graph ^
REM     --skip-drift ^
REM     --gpu ^
REM     --debug
REM echo Graph-based quick compression experiments completed!

echo All quick compression experiments completed! 