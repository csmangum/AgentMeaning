@echo off
REM Feature-Grouped Architecture Investigation
REM Tests the improved version of FeatureGroupedVAE that applies different compression levels
REM to different feature groups based on their importance.

echo Running Feature-Grouped Architecture Investigation...

REM Set Python path - update this if needed
set PYTHON=python

REM Create results directory if it doesn't exist
if not exist "meaning_transform\analysis\results" mkdir "meaning_transform\analysis\results"

REM Run Python script with custom feature-grouped architecture test
%PYTHON% -c "import sys; sys.path.append('.'); from meaning_transform.analysis.model_architecture_investigation import test_feature_grouped_architecture; test_feature_grouped_architecture()"

echo Feature-Grouped Architecture Investigation completed!
echo Check results in meaning_transform/analysis/results directory 