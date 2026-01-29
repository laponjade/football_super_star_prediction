"""
Test script to verify all 4 MLOps phases work correctly.
This script runs with reduced sweep trials (2) for faster testing.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_phase_1():
    """Test Phase 1: Data Versioning"""
    print("\n" + "=" * 70)
    print("TESTING PHASE 1: Data Versioning")
    print("=" * 70)
    
    try:
        from mlops_data_versioning import prepare_and_version_data
        
        print("Running Phase 1...")
        prepare_and_version_data()
        
        print("\n[SUCCESS] Phase 1 completed successfully!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Phase 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_2():
    """Test Phase 2: Experiment Tracking"""
    print("\n" + "=" * 70)
    print("TESTING PHASE 2: Experiment Tracking")
    print("=" * 70)
    
    try:
        from mlops_training_tracked import train_with_tracking
        
        print("Running Phase 2...")
        model, scaler = train_with_tracking()
        
        print("\n[SUCCESS] Phase 2 completed successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Scaler type: {type(scaler)}")
        return True
    except Exception as e:
        print(f"\n[FAIL] Phase 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_3(num_trials=2):
    """Test Phase 3: Hyperparameter Sweep (with reduced trials for testing)"""
    print("\n" + "=" * 70)
    print(f"TESTING PHASE 3: Hyperparameter Sweep ({num_trials} trials for testing)")
    print("=" * 70)
    
    try:
        from mlops_hyperparameter_sweep import run_sweep
        
        print(f"Running Phase 3 with {num_trials} trials (reduced for testing)...")
        print("   Note: Full sweep uses 20 trials, this test uses 2 for speed.")
        
        sweep_id = run_sweep(num_trials=num_trials)
        
        if sweep_id:
            print(f"\n[SUCCESS] Phase 3 completed successfully!")
            print(f"   Sweep ID: {sweep_id}")
            return sweep_id
        else:
            print("\n[FAIL] Phase 3 did not return sweep_id")
            return None
    except KeyboardInterrupt:
        print("\n[WARN] Phase 3 cancelled by user")
        return None
    except Exception as e:
        print(f"\n[FAIL] Phase 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_phase_4(sweep_id):
    """Test Phase 4: Model Registration"""
    print("\n" + "=" * 70)
    print("TESTING PHASE 4: Model Registration")
    print("=" * 70)
    
    if not sweep_id:
        print("\n[FAIL] No sweep_id provided. Phase 4 requires a completed sweep.")
        return False
    
    try:
        from mlops_model_registry import register_best_model
        
        print(f"Running Phase 4 with sweep_id: {sweep_id}")
        register_best_model(sweep_id=sweep_id)
        
        print("\n[SUCCESS] Phase 4 completed successfully!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Phase 4 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all 4 phases in sequence"""
    print("=" * 70)
    print("MLOPS PIPELINE - COMPLETE TEST")
    print("=" * 70)
    print("\nThis script will test all 4 phases:")
    print("  1. Phase 1: Data Versioning")
    print("  2. Phase 2: Experiment Tracking")
    print("  3. Phase 3: Hyperparameter Sweep (2 trials for testing)")
    print("  4. Phase 4: Model Registration")
    print("\nNote: Phase 3 uses 2 trials instead of 20 for faster testing.")
    print("      Full production runs should use 20 trials.")
    
    results = {}
    
    # Test Phase 1
    results['phase1'] = test_phase_1()
    if not results['phase1']:
        print("\n[ERROR] Phase 1 failed. Stopping tests.")
        return
    
    # Test Phase 2
    results['phase2'] = test_phase_2()
    if not results['phase2']:
        print("\n[ERROR] Phase 2 failed. Stopping tests.")
        return
    
    # Test Phase 3 (with reduced trials)
    sweep_id = test_phase_3(num_trials=2)
    results['phase3'] = sweep_id is not None
    if not results['phase3']:
        print("\n[ERROR] Phase 3 failed. Cannot test Phase 4.")
        return
    
    # Test Phase 4
    results['phase4'] = test_phase_4(sweep_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for phase, success in results.items():
        if phase == 'phase3' and isinstance(success, bool) and success:
            status = f"[OK] PASSED (sweep_id: {sweep_id})"
        else:
            status = "[OK] PASSED" if success else "[FAIL] FAILED"
        print(f"{phase.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[SUCCESS] All 4 phases completed successfully!")
        print("\nNext steps:")
        print("  1. Check W&B dashboard for all results")
        print("  2. Verify model registration: python check_model_registration.py")
        print("  3. For production, run Phase 3 with 20 trials (default)")
    else:
        print("\n[WARN] Some phases failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Tests interrupted by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
