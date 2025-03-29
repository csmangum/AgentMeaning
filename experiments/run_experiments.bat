@echo off
REM Script to run and manage experiments for the Meaning-Preserving Transformation System

setlocal enabledelayedexpansion

REM Default settings
set EXPERIMENTS_DIR=experiments
set RESULTS_DIR=results
set QUICK_MODE=0
set LIST_EXPERIMENTS=0
set CLEAN_RESULTS=0
set CREATE_REPORT=0
set EXPERIMENTS=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--list" (
    set LIST_EXPERIMENTS=1
    shift
    goto :parse_args
)
if /i "%~1"=="--clean" (
    set CLEAN_RESULTS=1
    shift
    goto :parse_args
)
if /i "%~1"=="--report" (
    set CREATE_REPORT=1
    shift
    goto :parse_args
)
if /i "%~1"=="--quick" (
    set QUICK_MODE=1
    shift
    goto :parse_args
)
if /i "%~1"=="--results-dir" (
    set RESULTS_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    echo Usage: run_experiments.bat [options] [experiment_names...]
    echo Options:
    echo   --list           List available experiments
    echo   --clean          Clean results directory
    echo   --report         Create summary report
    echo   --quick          Run experiments in quick mode
    echo   --results-dir    Specify custom results directory
    echo   --help           Show this help message
    echo.
    echo Examples:
    echo   run_experiments.bat --list
    echo   run_experiments.bat --quick compression_experiments
    echo   run_experiments.bat compression_experiments feature_importance_analysis
    exit /b 0
)

REM If not a known option, assume it's an experiment name
set EXPERIMENTS=!EXPERIMENTS! %1
shift
goto :parse_args
:end_parse_args

REM Create the results directory if it doesn't exist
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

REM List experiments
if %LIST_EXPERIMENTS%==1 (
    echo Available experiments:
    for %%F in (%EXPERIMENTS_DIR%\run_*.bat) do (
        set "filename=%%~nF"
        if "!filename:~0,10!" NEQ "run_quick_" (
            set "expname=!filename:~4!"
            echo - !expname!
            
            REM Check for quick version
            if exist "%EXPERIMENTS_DIR%\run_quick_!expname!.bat" (
                echo   ^(quick version available^)
            )
            
            REM Get description from second line
            for /f "skip=1 tokens=1*" %%A in ('type "%%F"') do (
                if "%%A"=="REM" echo   %%B
                goto :next_file
            )
            :next_file
        )
    )
    exit /b 0
)

REM Clean results directory
if %CLEAN_RESULTS%==1 (
    if exist "%RESULTS_DIR%" (
        echo Do you want to clean the results directory? This will delete all experiment results.
        set /p confirm=Type 'y' to confirm: 
        
        if /i "!confirm!"=="y" (
            rmdir /s /q "%RESULTS_DIR%"
            mkdir "%RESULTS_DIR%"
            echo Results directory cleaned.
        ) else (
            echo Operation cancelled.
        )
    ) else (
        echo Results directory doesn't exist. Nothing to clean.
    )
    exit /b 0
)

REM Create report
if %CREATE_REPORT%==1 (
    if exist "%RESULTS_DIR%" (
        echo Creating experiment summary report...
        
        REM Create the report directory if it doesn't exist
        if not exist "%RESULTS_DIR%\report" mkdir "%RESULTS_DIR%\report"
        
        REM Create a simple report file
        set REPORT_FILE=%RESULTS_DIR%\report\experiment_summary.txt
        echo Experiment Summary Report > "%REPORT_FILE%"
        echo Generated: %date% %time% >> "%REPORT_FILE%"
        echo. >> "%REPORT_FILE%"
        echo Overview >> "%REPORT_FILE%"
        echo ======== >> "%REPORT_FILE%"
        echo This report summarizes the results of experiments run on the Meaning-Preserving Transformation System. >> "%REPORT_FILE%"
        echo. >> "%REPORT_FILE%"
        echo Experiment Results >> "%REPORT_FILE%"
        echo ================= >> "%REPORT_FILE%"
        
        REM List experiment directories and configs
        for /d %%D in (%RESULTS_DIR%\*) do (
            if "%%~nxD" NEQ "report" (
                echo. >> "%REPORT_FILE%"
                echo %%~nxD >> "%REPORT_FILE%"
                echo -------------------------- >> "%REPORT_FILE%"
                
                REM Check for config.json
                if exist "%%D\config.json" (
                    echo Configuration: >> "%REPORT_FILE%"
                    type "%%D\config.json" >> "%REPORT_FILE%"
                    echo. >> "%REPORT_FILE%"
                )
                
                REM List metrics files
                echo Metrics Files: >> "%REPORT_FILE%"
                for /r "%%D" %%F in (*metrics*.json) do (
                    echo - %%~nxF >> "%REPORT_FILE%"
                )
                echo. >> "%REPORT_FILE%"
                
                REM List charts (simple text listing, can't embed images in txt)
                echo Charts: >> "%REPORT_FILE%"
                for /r "%%D" %%F in (*.png) do (
                    echo - %%~nxF >> "%REPORT_FILE%"
                )
                echo. >> "%REPORT_FILE%"
            )
        )
        
        echo Report created at: %REPORT_FILE%
        
        REM Try to use PowerShell for a better report if available
        powershell -Command "& {if (Get-Command 'powershell' -ErrorAction SilentlyContinue) { & '.\experiments\run_experiments.ps1' -CreateReport -ResultsDir '%RESULTS_DIR%' }}"
    ) else (
        echo Results directory doesn't exist. No report to create.
    )
    exit /b 0
)

REM Check if experiments were specified
if "%EXPERIMENTS%"=="" (
    echo No experiments specified. Use run_experiments.bat [experiment_names...] to run experiments.
    echo Use run_experiments.bat --list to see available experiments.
    exit /b 1
)

REM Run specified experiments
for %%E in (%EXPERIMENTS%) do (
    set EXPERIMENT=%%E
    set BAT_FILE=
    
    if %QUICK_MODE%==1 (
        if exist "%EXPERIMENTS_DIR%\run_quick_!EXPERIMENT!.bat" (
            set BAT_FILE=%EXPERIMENTS_DIR%\run_quick_!EXPERIMENT!.bat
        ) else (
            echo Quick version of !EXPERIMENT! not found. Falling back to regular version.
        )
    )
    
    if "!BAT_FILE!"=="" (
        if exist "%EXPERIMENTS_DIR%\run_!EXPERIMENT!.bat" (
            set BAT_FILE=%EXPERIMENTS_DIR%\run_!EXPERIMENT!.bat
        ) else (
            echo Experiment !EXPERIMENT! not found. Skipping.
            goto :next_experiment
        )
    )
    
    echo Running experiment: !EXPERIMENT! !BAT_FILE!
    call !BAT_FILE!
    echo Experiment completed: !EXPERIMENT!
    echo.
    
    :next_experiment
)

REM Offer to create a report if multiple experiments were run
set /a COUNT=0
for %%E in (%EXPERIMENTS%) do set /a COUNT+=1

if %COUNT% GTR 1 (
    echo Multiple experiments completed. Do you want to generate a summary report?
    set /p create_report=Type 'y' to create a report: 
    
    if /i "!create_report!"=="y" (
        call %0 --report
    )
)

echo All experiments completed!
exit /b 0 