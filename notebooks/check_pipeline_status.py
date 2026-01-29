"""Quick script to check MLOps pipeline completion status"""
import json
from pathlib import Path
from datetime import datetime

def check_latest_runs():
    """Check the most recent W&B runs to see what completed"""
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        print("No wandb directory found")
        return
    
    # Find most recent runs
    runs = []
    for run_dir in wandb_dir.glob("run-*"):
        if run_dir.is_dir():
            summary_file = run_dir / "files" / "wandb-summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    runs.append({
                        'name': run_dir.name,
                        'summary': summary,
                        'mtime': summary_file.stat().st_mtime
                    })
                except:
                    pass
    
    # Sort by modification time
    runs.sort(key=lambda x: x['mtime'], reverse=True)
    
    print("=" * 60)
    print("RECENT MLOPS RUNS STATUS")
    print("=" * 60)
    
    # Check Phase 1 (Data Versioning)
    phase1_runs = [r for r in runs if 'dataset/train_size' in r['summary']]
    if phase1_runs:
        latest = phase1_runs[0]
        print(f"\n[OK] Phase 1 (Data Versioning) - Latest run: {latest['name']}")
        print(f"     Train size: {latest['summary'].get('dataset/train_size', 'N/A')}")
        print(f"     Val size: {latest['summary'].get('dataset/val_size', 'N/A')}")
    
    # Check Phase 2 (Training)
    phase2_runs = [r for r in runs if 'val/f1' in r['summary']]
    if phase2_runs:
        latest = phase2_runs[0]
        print(f"\n[OK] Phase 2 (Experiment Tracking) - Latest run: {latest['name']}")
        print(f"     Val F1: {latest['summary'].get('val/f1', 'N/A'):.4f}")
        print(f"     Val Accuracy: {latest['summary'].get('val/accuracy', 'N/A'):.4f}")
        print(f"     Test F1: {latest['summary'].get('test/f1', 'N/A'):.4f}")
    
    # Check Phase 3 (Sweep)
    sweep_dirs = list(wandb_dir.glob("sweep-*"))
    if sweep_dirs:
        print(f"\n[OK] Phase 3 (Hyperparameter Sweep) - Found {len(sweep_dirs)} sweep(s)")
        for sweep_dir in sweep_dirs[:3]:  # Show first 3
            print(f"     Sweep ID: {sweep_dir.name}")
    
    # Check Phase 4 (Model Registry)
    phase4_runs = [r for r in runs if 'model_registry' in str(r['summary']).lower() or 'registered' in str(r['summary']).lower()]
    if phase4_runs:
        print(f"\n[OK] Phase 4 (Model Registry) - Found registered model")
    else:
        print(f"\n[INFO] Phase 4 (Model Registry) - No registered model found yet")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_latest_runs()
