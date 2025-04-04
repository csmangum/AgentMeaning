@echo off
REM Feature Importance Hierarchy Robustness Analysis
REM Implementation of Step 18 in the project roadmap

REM Create output directory
mkdir findings\feature_importance_robustness 2>nul

REM Run the analysis using our dedicated Python script
python experiments\feature_importance_analysis.py

echo Feature Importance Hierarchy Robustness Analysis completed!
echo Results available in the findings\feature_importance_robustness directory 