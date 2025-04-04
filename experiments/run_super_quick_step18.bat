@echo off
REM Super Quick Feature Importance Hierarchy Robustness Analysis
REM Minimal implementation that just analyzes feature importance without model training

REM Create output directory
mkdir findings\feature_importance_robustness_super_quick 2>nul

REM Run the simplified analysis 
python experiments\quick_feature_importance.py

echo Super Quick Feature Importance Hierarchy Robustness Analysis completed!
echo Results available in the findings\feature_importance_robustness_super_quick directory 