param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("control_current", "o1_muon_adamw", "o2_wsd", "a1_depth_width_12x640", "a1_depth_width_16x576", "r1_fixed_1024")]
    [string]$Experiment,

    [ValidateSet("quick", "full")]
    [string]$Profile = "quick",

    [string]$EvalData = "provided_val.bin"
)

$ErrorActionPreference = "Stop"

$configPath = "config/tier1_ablation/$Experiment.py"
if (-not (Test-Path $configPath)) {
    throw "Config not found: $configPath"
}

$profileOverrides = @()
if ($Profile -eq "quick") {
    $profileOverrides = @(
        "max_iters=2000",
        "lr_decay_iters=2000",
        "eval_interval=200",
        "eval_iters=25",
        "out_dir=out_tier1_quick_$Experiment"
    )
}

Write-Host "==> Running $Experiment ($Profile)"
python train.py $configPath @profileOverrides

$outDir = ""
if ($Profile -eq "quick") {
    $outDir = "out_tier1_quick_$Experiment"
} else {
    # Keep this in sync with each config's out_dir.
    $outDirMap = @{
        "control_current"       = "out_tier1_control_current"
        "o1_muon_adamw"         = "out_tier1_o1_muon_adamw"
        "o2_wsd"                = "out_tier1_o2_wsd"
        "a1_depth_width_12x640" = "out_tier1_a1_12x640"
        "a1_depth_width_16x576" = "out_tier1_a1_16x576"
        "r1_fixed_1024"         = "out_tier1_r1_fixed1024"
    }
    $outDir = $outDirMap[$Experiment]
}

$checkpointPath = Join-Path $outDir "checkpoint.pt"
if (-not (Test-Path $checkpointPath)) {
    throw "Expected checkpoint not found: $checkpointPath"
}

Write-Host "==> Evaluating $checkpointPath on $EvalData"
python evaluate.py --model_dir . --checkpoint_filename $checkpointPath --data $EvalData
