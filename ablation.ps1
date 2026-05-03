$exps = @(
  "control_current",
  "o1_muon_adamw",
  "o2_wsd",
  "a1_depth_width_12x640",
  "a1_depth_width_16x576",
  "r1_fixed_1024"
)
foreach ($e in $exps) {
  powershell -ExecutionPolicy Bypass -File scripts/run_tier1_ablation.ps1 `
    -Experiment $e -Profile quick -EvalData provided_val.bin
}