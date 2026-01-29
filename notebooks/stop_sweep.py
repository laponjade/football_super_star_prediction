"""Utility to stop a running W&B sweep"""
import wandb
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stop_sweep.py <sweep_id>")
        print("\nTo find your sweep_id:")
        print("1. Check the W&B dashboard")
        print("2. Or look in wandb/sweep-*/ directory")
        sys.exit(1)
    
    sweep_id = sys.argv[1]
    print(f"Attempting to stop sweep: {sweep_id}")
    
    try:
        # Cancel the sweep
        api = wandb.Api()
        sweep = api.sweep(f"abdoubendaia7-cole-sup-rieure-en-informatique-sidi-bel-abbes/football-superstar-prediction/{sweep_id}")
        sweep.cancel()
        print(f"[OK] Sweep {sweep_id} cancelled successfully")
    except Exception as e:
        print(f"[FAIL] Could not stop sweep: {e}")
        print("\nAlternative: Press Ctrl+C in the terminal running the sweep")
        print("It will finish the current trial and then stop.")
