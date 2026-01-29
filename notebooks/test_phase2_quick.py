"""Quick test of Phase 2 to identify issues"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 60)
print("TESTING PHASE 2 - QUICK CHECK")
print("=" * 60)

try:
    print("\n1. Testing imports...")
    from mlops_training_tracked import train_with_tracking
    from mlops_config import ENTITY, PROJECT
    print(f"   [OK] Imports successful")
    print(f"   [INFO] Entity: {ENTITY}")
    print(f"   [INFO] Project: {PROJECT}")
except Exception as e:
    print(f"   [FAIL] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. Testing W&B initialization...")
    import wandb
    print(f"   [OK] W&B version: {wandb.__version__}")
    print(f"   [INFO] W&B login status will be checked when run starts")
except Exception as e:
    print(f"   [FAIL] W&B error: {e}")
    sys.exit(1)

try:
    print("\n3. Testing data file access...")
    from mlops_config import DATA_PATH
    data_path = Path(DATA_PATH)
    if data_path.exists():
        print(f"   [OK] Data file exists: {data_path}")
        print(f"   [INFO] File size: {data_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"   [FAIL] Data file not found: {data_path}")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Data file check error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - Phase 2 should work!")
print("=" * 60)
print("\nTo run Phase 2:")
print("  python mlops_training_tracked.py")
print("\nOr via pipeline:")
print("  python run_mlops_pipeline.py")
print("  (Choose option 2)")
