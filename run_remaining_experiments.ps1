# Save this script in the root of fruit-illum-norm.
# Run from PowerShell after: conda activate fruit-gpu

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ($env:CONDA_DEFAULT_ENV -ne 'fruit-gpu') {
    throw 'Activate the fruit-gpu Conda environment first: conda activate fruit-gpu'
}

# Avoid accidentally using the simultaneously activated .venv interpreter.
$python = Join-Path $env:CONDA_PREFIX 'python.exe'
if (!(Test-Path $python)) { throw "Python not found: $python" }

& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('Training GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed. Training stopped before starting.' }

$logDir = Join-Path $PSScriptRoot 'outputs\automation_logs'
$markerDir = Join-Path $logDir 'completed'
New-Item -ItemType Directory -Force -Path $logDir, $markerDir | Out-Null

# E4 can be added later, after deciding to include the optional experiment.
$conditions = @('E2', 'E3')

function Invoke-Step {
    param(
        [string]$Condition,
        [string]$Step,
        [string]$OutputFile,
        [string[]]$Arguments
    )

    $marker = Join-Path $markerDir "${Condition}_${Step}.done"
    $outputPath = Join-Path $PSScriptRoot $OutputFile
    if ((Test-Path $marker) -and (Test-Path $outputPath)) {
        Write-Host "SKIP: $Condition $Step (already completed)" -ForegroundColor Cyan
        return
    }

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logPath = Join-Path $logDir "${Condition}_${Step}_${timestamp}.log"
    Write-Host "START: $Condition $Step" -ForegroundColor Yellow
    Write-Host "Log: $logPath"

    Start-Transcript -Path $logPath | Out-Null
    try {
        & $python @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Stop-Transcript | Out-Null
    }

    if ($exitCode -ne 0) {
        throw "$Condition $Step failed with exit code $exitCode. See $logPath"
    }
    if (!(Test-Path $outputPath)) {
        throw "$Condition $Step ended but expected output was not found: $outputPath"
    }

    # Mark success only after the subprocess finishes and output exists.
    "Completed $(Get-Date -Format o)" | Set-Content -Path $marker
    Write-Host "DONE: $Condition $Step" -ForegroundColor Green
}

foreach ($condition in $conditions) {
    Invoke-Step -Condition $condition -Step 'stage1' `
        -OutputFile "outputs\checkpoints\${condition}_seed42_best.pth" `
        -Arguments @('-m','src.train','--condition',$condition,'--stage','1',
                     '--seed','42','--epochs','30','--batch-size','16',
                     '--lr','1e-4','--patience','4','--wandb-mode','online')

    Invoke-Step -Condition $condition -Step 'stage2' `
        -OutputFile "outputs\checkpoints\${condition}_seed42_stage2_best.pth" `
        -Arguments @('-m','src.train','--condition',$condition,'--stage','2',
                     '--seed','42','--epochs','40','--batch-size','16',
                     '--lr','1e-5','--patience','4','--wandb-mode','online')

    Invoke-Step -Condition $condition -Step 'evaluation' `
        -OutputFile "outputs\metrics\${condition}_seed42_stage2_normal_test_metrics.json" `
        -Arguments @('-m','src.evaluate','--condition',$condition,'--stage','2',
                     '--seed','42','--stress','normal','--wandb-mode','online')
}

Write-Host 'All planned E2 and E3 experiments completed successfully!' -ForegroundColor Green
