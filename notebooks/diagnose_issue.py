"""Diagnose what's not working in the MLOps pipeline"""
import json
from pathlib import Path

print("=" * 60)
print("MLOPS PIPELINE DIAGNOSTIC")
print("=" * 60)

# Check latest run
wandb_dir = Path("wandb")
if wandb_dir.exists():
    runs = sorted([d for d in wandb_dir.glob("run-*") if d.is_dir()], 
                   key=lambda p: p.stat().st_mtime, reverse=True)
    
    if runs:
        latest_run = runs[0]
        print(f"\nLatest run: {latest_run.name}")
        
        summary_file = latest_run / "files" / "wandb-summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                print("\nMetrics found in latest run:")
                self_training_metrics = [k for k in summary.keys() if 'self_training' in k]
                val_metrics = [k for k in summary.keys() if k.startswith('val/')]
                test_metrics = [k for k in summary.keys() if k.startswith('test/')]
                
                if self_training_metrics:
                    print(f"  [OK] Self-training metrics: {len(self_training_metrics)} found")
                    print(f"       Examples: {self_training_metrics[:3]}")
                else:
                    print(f"  [WARN] No self-training metrics found")
                
                if val_metrics:
                    print(f"  [OK] Validation metrics: {len(val_metrics)} found")
                    print(f"       Examples: {val_metrics[:3]}")
                else:
                    print(f"  [WARN] No validation metrics found")
                
                if test_metrics:
                    print(f"  [OK] Test metrics: {len(test_metrics)} found")
                else:
                    print(f"  [WARN] No test metrics found")
                
                # Check for the time-series metric
                if 'self_training/labels_added' in summary or any('labels_added' in k for k in summary.keys()):
                    print(f"  [OK] Found labels_added metric (should show time-series chart)")
                else:
                    print(f"  [WARN] labels_added metric not found in summary (may be in history)")
                
            except Exception as e:
                print(f"  [ERROR] Could not read summary: {e}")
    else:
        print("\n[WARN] No runs found in wandb directory")
else:
    print("\n[WARN] wandb directory not found")

# Check code
print("\n" + "=" * 60)
print("CODE CHECK")
print("=" * 60)

training_file = Path("mlops_training_tracked.py")
if training_file.exists():
    content = training_file.read_text()
    if 'self_training/labels_added' in content:
        print("  [OK] Code has self_training/labels_added metric")
    else:
        print("  [FAIL] Code missing self_training/labels_added metric")
    
    if 'step=i+1' in content:
        print("  [OK] Code uses step numbers for logging")
    else:
        print("  [WARN] Code may not use step numbers properly")
else:
    print("  [FAIL] mlops_training_tracked.py not found")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("1. Check W&B dashboard: https://wandb.ai/abdoubendaia7-cole-sup-rieure-en-informatique-sidi-bel-abbes/football-superstar-prediction")
print("2. Look for chart: 'self_training/labels_added' in the metrics section")
print("3. If charts are empty, the run may have logged metrics but W&B needs time to process")
print("4. Try running Phase 2 again: python mlops_training_tracked.py")
