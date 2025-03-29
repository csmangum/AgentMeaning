# PowerShell script to run and manage experiments for the Meaning-Preserving Transformation System
param (
    [Parameter(Mandatory=$false)]
    [string[]]$Experiments,

    [Parameter(Mandatory=$false)]
    [switch]$QuickMode,

    [Parameter(Mandatory=$false)]
    [switch]$ListExperiments,

    [Parameter(Mandatory=$false)]
    [switch]$CleanResults,

    [Parameter(Mandatory=$false)]
    [string]$ResultsDir,

    [Parameter(Mandatory=$false)]
    [switch]$CreateReport
)

# Directory settings
$EXPERIMENTS_DIR = "experiments"
$RESULTS_DIR = if ($ResultsDir) { $ResultsDir } else { "results" }

# Function to list available experiments
function List-Experiments {
    Write-Host "Available experiments:"
    $batFiles = Get-ChildItem -Path $EXPERIMENTS_DIR -Filter "run_*.bat" | Where-Object { $_.Name -notlike "run_quick_*" } | Sort-Object Name
    
    foreach ($file in $batFiles) {
        $experimentName = $file.Name -replace "run_", "" -replace ".bat", ""
        $firstLine = Get-Content $file.FullName -TotalCount 2 | Select-Object -Last 1
        $description = $firstLine -replace "REM ", ""
        
        $hasQuick = Test-Path (Join-Path $EXPERIMENTS_DIR "run_quick_$experimentName.bat")
        $quickInfo = if ($hasQuick) { " (quick version available)" } else { "" }
        
        Write-Host "- $experimentName$quickInfo"
        Write-Host "  $description"
    }
}

# Function to clean results directory
function Clean-Results {
    if (-not (Test-Path $RESULTS_DIR)) {
        Write-Host "Results directory doesn't exist. Nothing to clean."
        return
    }
    
    Write-Host "Do you want to clean the results directory? This will delete all experiment results. (y/n)"
    $confirm = Read-Host
    
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        Remove-Item -Path $RESULTS_DIR -Recurse -Force
        Write-Host "Results directory cleaned."
        New-Item -Path $RESULTS_DIR -ItemType Directory | Out-Null
    } else {
        Write-Host "Operation cancelled."
    }
}

# Function to create a summary report of experiment results
function Create-Report {
    if (-not (Test-Path $RESULTS_DIR)) {
        Write-Host "Results directory doesn't exist. No report to create."
        return
    }
    
    $reportDir = Join-Path $RESULTS_DIR "report"
    if (-not (Test-Path $reportDir)) {
        New-Item -Path $reportDir -ItemType Directory | Out-Null
    }
    
    $reportPath = Join-Path $reportDir "experiment_summary.md"
    
    # Start writing the report
    $reportContent = @"
# Experiment Summary Report
Generated: $(Get-Date)

## Overview

This report summarizes the results of experiments run on the Meaning-Preserving Transformation System.

## Experiment Results

"@
    
    # Get all experiment directories
    $experimentDirs = Get-ChildItem -Path $RESULTS_DIR -Directory | Where-Object { $_.Name -ne "report" }
    
    foreach ($dir in $experimentDirs) {
        $reportContent += @"

### $($dir.Name)

"@
        
        # Check if config.json exists
        $configPath = Join-Path $dir.FullName "config.json"
        if (Test-Path $configPath) {
            $config = Get-Content $configPath | ConvertFrom-Json
            $reportContent += "**Configuration:**`n`n```json`n$(Get-Content $configPath -Raw)`n````n`n"
        }
        
        # Look for metrics or results files
        $metricsFiles = Get-ChildItem -Path $dir.FullName -Filter "*metrics*.json" -Recurse
        foreach ($file in $metricsFiles) {
            $reportContent += "**$(Split-Path $file.FullName -Leaf):**`n`n```json`n$(Get-Content $file.FullName -Raw)`n````n`n"
        }
        
        # Look for any .png files (charts)
        $chartFiles = Get-ChildItem -Path $dir.FullName -Filter "*.png" -Recurse
        if ($chartFiles.Count -gt 0) {
            $reportContent += "**Charts:**`n`n"
            foreach ($chart in $chartFiles) {
                $relativePath = $chart.FullName.Replace("$PWD\", "").Replace("\", "/")
                $reportContent += "![$(Split-Path $chart.FullName -Leaf)]($relativePath)`n`n"
            }
        }
    }
    
    # Write the report file
    $reportContent | Out-File -FilePath $reportPath -Encoding utf8
    
    Write-Host "Report generated at: $reportPath"
}

# Function to run a specific experiment
function Run-Experiment {
    param (
        [string]$ExperimentName,
        [switch]$Quick
    )
    
    $batFileName = if ($Quick) { "run_quick_$ExperimentName.bat" } else { "run_$ExperimentName.bat" }
    $batFilePath = Join-Path $EXPERIMENTS_DIR $batFileName
    
    if (Test-Path $batFilePath) {
        Write-Host "Running experiment: $ExperimentName $(if ($Quick) { '(quick mode)' } else { '' })"
        $startTime = Get-Date
        
        # Run the batch file
        & cmd.exe /c $batFilePath
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Host "Experiment completed in $($duration.TotalSeconds) seconds."
    } else {
        Write-Host "Experiment batch file not found: $batFilePath"
        if ($Quick) {
            Write-Host "Quick version not available. Try running without -QuickMode."
        }
    }
}

# Main execution
if ($ListExperiments) {
    List-Experiments
    exit 0
}

if ($CleanResults) {
    Clean-Results
    exit 0
}

if ($CreateReport) {
    Create-Report
    exit 0
}

if (-not $Experiments -or $Experiments.Count -eq 0) {
    Write-Host "No experiments specified. Use -Experiments parameter to specify which experiments to run."
    Write-Host "Use -ListExperiments to see available experiments."
    exit 1
}

# Create results directory if it doesn't exist
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -Path $RESULTS_DIR -ItemType Directory | Out-Null
}

# Run specified experiments
foreach ($exp in $Experiments) {
    Run-Experiment -ExperimentName $exp -Quick:$QuickMode
}

# Offer to create a report
if ($Experiments.Count -gt 1) {
    Write-Host "Multiple experiments completed. Do you want to generate a summary report? (y/n)"
    $createReportAnswer = Read-Host
    
    if ($createReportAnswer -eq "y" -or $createReportAnswer -eq "Y") {
        Create-Report
    }
} 