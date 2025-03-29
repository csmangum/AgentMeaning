@echo off
REM Model Architecture Investigation for Meaning-Preserving Transformations
REM Step 13: Analysis of model architecture and size related to compression levels
REM (Fixed version with device handling, FeatureGroupedVAE compatibility, tensor size and recursion fixes)

echo Running Model Architecture Investigation (Fixed Version 2.0)...

REM Set Python path - update this if needed
set PYTHON=python

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "meaning_transform\analysis\results" mkdir "meaning_transform\analysis\results"

REM Run full architecture investigation with fixes
%PYTHON% meaning_transform/analysis/model_architecture_investigation.py ^
    --db-path "simulation.db" ^
    --epochs 15 ^
    --gpu

echo Model Architecture Investigation completed!
echo Check results in meaning_transform/analysis/results directory 