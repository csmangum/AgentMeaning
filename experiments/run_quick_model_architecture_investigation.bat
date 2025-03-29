@echo off
REM Quick Model Architecture Investigation (using mock models)
REM Step 13: Analysis of model architecture and size related to compression levels
REM (Fixed version with device handling, FeatureGroupedVAE compatibility, and improved feature-grouped compression)

echo Running Quick Model Architecture Investigation (Improved FeatureGrouped Version)...

REM Set Python path - update this if needed
set PYTHON=python

REM Create results directory if it doesn't exist
if not exist "meaning_transform\analysis\results" mkdir "meaning_transform\analysis\results"

REM Run quick architecture investigation using mock models
%PYTHON% meaning_transform/analysis/model_architecture_investigation.py ^
    --quick

echo Quick Model Architecture Investigation completed!
echo Check results in meaning_transform/analysis/results directory 