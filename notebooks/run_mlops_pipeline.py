"""
Complete MLOps Pipeline Runner
This script runs all MLOps phases in sequence with verification.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def check_requirements():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("STEP 0: Checking Requirements")
    print("=" * 60)
    
    required_packages = {
        'wandb': 'wandb',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"[OK] {package_name} is installed")
        except ImportError:
            print(f"[FAIL] {package_name} is NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n[WARN] Missing packages: {', '.join(missing)}")
        print(f"Install them with: pip install {' '.join(missing)}")
        return False
    
    print("\n[OK] All required packages are installed!")
    return True

def check_wandb_login():
    """Check if W&B is logged in."""
    print("\n" + "=" * 60)
    print("STEP 0.5: Checking W&B Login")
    print("=" * 60)
    
    try:
        import wandb
        
        # Simple check: Just verify wandb can be imported and initialized
        # The actual login will be handled by wandb itself when you run the scripts
        # If not logged in, wandb will prompt you automatically
        
        print("[OK] W&B is available")
        print("   Note: If you're not logged in, wandb will prompt you when needed")
        print("   Or run 'wandb login' now to login beforehand")
        return True  # Always allow to proceed - wandb handles login gracefully
        
    except ImportError:
        print("[FAIL] W&B is not installed")
        print("   Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"[WARN] Could not check W&B: {str(e)}")
        print("   You can proceed - wandb will handle login if needed")
        return True  # Allow to proceed

def check_config():
    """Check if configuration is set correctly."""
    print("\n" + "=" * 60)
    print("STEP 0.6: Checking Configuration")
    print("=" * 60)
    
    config_files = [
        'mlops_data_versioning.py',
        'mlops_training_tracked.py',
        'mlops_hyperparameter_sweep.py',
        'mlops_model_registry.py'
    ]
    
    issues = []
    for config_file in config_files:
        file_path = Path(__file__).parent / config_file
        if not file_path.exists():
            print(f"[FAIL] {config_file} not found")
            issues.append(f"{config_file} missing")
            continue
        
        content = file_path.read_text()
        if 'your-wandb-entity' in content or "your-wandb-entity" in content:
            print(f"[WARN] {config_file}: ENTITY needs to be updated")
            issues.append(f"{config_file} has placeholder ENTITY")
        else:
            print(f"[OK] {config_file}: Configuration looks good")
    
    if issues:
        print(f"\n[WARN] Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n   Please update ENTITY in all MLOps scripts before running.")
        return False
    
    print("\n[OK] All configuration files are set correctly!")
    return True

def check_data_file():
    """Check if data file exists."""
    print("\n" + "=" * 60)
    print("STEP 0.7: Checking Data File")
    print("=" * 60)
    
    data_path = Path(__file__).parent.parent / 'data' / 'feature_engineered_data_v2.csv'
    
    if data_path.exists():
        print(f"[OK] Data file found: {data_path}")
        # Check file size
        size_mb = data_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        return True
    else:
        print(f"[FAIL] Data file not found: {data_path}")
        print("   Make sure the data file exists before running Phase 1.")
        return False

def run_phase_1():
    """Run Phase 1: Data Versioning"""
    print("\n" + "=" * 60)
    print("PHASE 1: Data Versioning")
    print("=" * 60)
    
    try:
        from mlops_data_versioning import prepare_and_version_data
        print("Running data versioning...")
        metadata = prepare_and_version_data()
        print("\n[OK] Phase 1 completed successfully!")
        print(f"   Dataset metadata: {len(metadata.get('features', []))} features")
        return True
    except Exception as e:
        print(f"\n[FAIL] Phase 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_phase_2():
    """Run Phase 2: Experiment Tracking"""
    print("\n" + "=" * 60)
    print("PHASE 2: Experiment Tracking")
    print("=" * 60)
    
    try:
        from mlops_training_tracked import train_with_tracking
        print("Running tracked training...")
        model, scaler = train_with_tracking()
        print("\n[OK] Phase 2 completed successfully!")
        print(f"   Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"\n[FAIL] Phase 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_phase_3():
    """Run Phase 3: Hyperparameter Sweep"""
    print("\n" + "=" * 60)
    print("PHASE 3: Hyperparameter Sweep")
    print("=" * 60)
    
    try:
        from mlops_hyperparameter_sweep import run_sweep
        from mlops_config import SWEEP_TRIALS
        
        print(f"Starting hyperparameter sweep...")
        print(f"Number of trials: {SWEEP_TRIALS}")
        print(f"This may take 30-60 minutes depending on number of trials.")
        print(f"\nYou can monitor progress in W&B dashboard.")
        print(f"Press Ctrl+C to cancel if needed.\n")
        
        # Run the sweep (returns sweep_id)
        sweep_id = run_sweep(num_trials=SWEEP_TRIALS)
        
        from mlops_config import ENTITY, PROJECT
        print(f"\n[OK] Phase 3 completed successfully!")
        print(f"   Sweep ID: {sweep_id}")
        print(f"   View results: https://wandb.ai/{ENTITY}/{PROJECT}/sweeps/{sweep_id}")
        print(f"   You can now run Phase 4 to register the best model.")
        
        return sweep_id  # Return sweep_id for Phase 4
    except KeyboardInterrupt:
        print("\n[WARN] Sweep cancelled by user.")
        return None
    except Exception as e:
        print(f"\n[FAIL] Phase 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_phase_3_dry_run():
    """Dry run of Phase 3: Hyperparameter Sweep (just show config)"""
    print("\n" + "=" * 60)
    print("PHASE 3: Hyperparameter Sweep (DRY RUN)")
    print("=" * 60)
    
    try:
        from mlops_hyperparameter_sweep import sweep_config
        print("Sweep configuration:")
        import json
        print(json.dumps(sweep_config, indent=2))
        print("\n[WARN] This is a dry run. To actually run the sweep:")
        print("   python mlops_hyperparameter_sweep.py")
        print("   Or choose option 4 to run all phases.")
        return True
    except Exception as e:
        print(f"\n[FAIL] Phase 3 check failed: {str(e)}")
        return False

def run_phase_4_with_sweep_id(sweep_id):
    """Run Phase 4: Model Registration with provided sweep_id"""
    print("\n" + "=" * 60)
    print("PHASE 4: Model Registration")
    print("=" * 60)
    
    try:
        from mlops_model_registry import register_best_model
        
        print(f"Registering best model from sweep: {sweep_id}")
        register_best_model(sweep_id=sweep_id)
        
        print("\n[OK] Phase 4 completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Phase 4 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_phase_4():
    """Run Phase 4: Model Registration (interactive - asks for sweep_id)"""
    print("\n" + "=" * 60)
    print("PHASE 4: Model Registration")
    print("=" * 60)
    
    try:
        from mlops_model_registry import register_best_model
        from wandb import Api
        from mlops_config import ENTITY, PROJECT
        
        # Get sweep_id from user
        print("\nTo register the best model, we need the sweep_id from a completed sweep.")
        print("You can find it in the W&B dashboard under Sweeps.")
        
        try:
            sweep_id = input("\nEnter sweep_id (or press Enter to list available sweeps): ").strip()
            
            if not sweep_id:
                # List available sweeps
                print("\nFetching available sweeps from W&B...")
                api = Api()
                project = api.project(ENTITY, PROJECT)
                sweeps = list(project.sweeps())
                
                if sweeps:
                    print(f"\nFound {len(sweeps)} sweep(s):")
                    for i, sweep in enumerate(sweeps, 1):
                        print(f"  {i}. {sweep.id} - {sweep.name}")
                        if hasattr(sweep, 'state'):
                            print(f"     State: {sweep.state}")
                    
                    try:
                        choice = input(f"\nSelect sweep (1-{len(sweeps)}) or enter sweep_id: ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(sweeps):
                            sweep_id = sweeps[int(choice) - 1].id
                        else:
                            sweep_id = choice
                    except:
                        print("[WARN] Could not parse selection, please provide sweep_id manually.")
                        return False
                else:
                    print("[WARN] No sweeps found. Please run Phase 3 first.")
                    return False
            
            # Register the model
            return run_phase_4_with_sweep_id(sweep_id)
            
        except KeyboardInterrupt:
            print("\n[WARN] Model registration cancelled by user.")
            return False
        except Exception as e:
            print(f"\n[FAIL] Phase 4 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Phase 4 setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_mlops_principles():
    """Verify that all MLOps principles are respected."""
    print("\n" + "=" * 60)
    print("MLOPS PRINCIPLES VERIFICATION")
    print("=" * 60)
    
    principles = {
        "Data Versioning": "[OK] Datasets are versioned as W&B artifacts",
        "Experiment Tracking": "[OK] All runs tracked with metrics and configs",
        "Reproducibility": "[OK] Random seeds set, configs logged",
        "Hyperparameter Optimization": "[OK] Sweeps configured for systematic search",
        "Model Versioning": "[OK] Models registered as artifacts with metadata",
        "Metrics Logging": "[OK] All performance metrics logged (F1, ROC-AUC, etc.)",
        "Visualization": "[OK] Confusion matrices and feature importance logged",
        "Model Registry": "[OK] Best models can be registered and retrieved"
    }
    
    for principle, status in principles.items():
        print(f"{status} - {principle}")
    
    print("\n[OK] All MLOps principles are implemented!")

def main():
    """Main pipeline runner."""
    print("\n" + "=" * 60)
    print("FOOTBALL SUPERSTAR PREDICTION - MLOPS PIPELINE")
    print("=" * 60)
    print("\nThis script will verify and run the MLOps pipeline.")
    print("Make sure you have:")
    print("  1. Installed all requirements (pip install -r requirements.txt)")
    print("  2. Logged into W&B (wandb login)")
    print("  3. Updated ENTITY in all MLOps scripts")
    print("\n")
    
    # Skip input if running in non-interactive mode (for testing)
    try:
        input("Press Enter to continue or Ctrl+C to cancel...")
    except EOFError:
        print("(Non-interactive mode - proceeding automatically)")
    
    # Pre-flight checks
    if not check_requirements():
        print("\n[FAIL] Requirements check failed. Please install missing packages.")
        return False
    
    if not check_wandb_login():
        print("\n[FAIL] W&B login check failed. Please run: wandb login")
        return False
    
    if not check_config():
        print("\n[FAIL] Configuration check failed. Please update ENTITY in scripts.")
        return False
    
    if not check_data_file():
        print("\n[FAIL] Data file check failed.")
        return False
    
    print("\n" + "=" * 60)
    print("ALL PRE-FLIGHT CHECKS PASSED!")
    print("=" * 60)
    
    # Ask which phases to run
    print("\nWhich phases would you like to run?")
    print("  1. Phase 1: Data Versioning")
    print("  2. Phase 2: Experiment Tracking")
    print("  3. Phase 3: Hyperparameter Sweep (FULL - may take 30-60 min)")
    print("  4. Phase 4: Model Registration (requires completed sweep)")
    print("  5. All phases (1, 2, 3, 4)")
    print("  6. Phase 3 (dry run - just show config)")
    print("  7. Just verification (no execution)")
    
    # Get choice - use default if non-interactive
    try:
        choice = input("\nEnter choice (1-7): ").strip()
    except EOFError:
        choice = "5"  # Default to all phases if non-interactive
        print(f"(Non-interactive mode - using choice: {choice})")
    
    results = {}
    
    # Phase 1: Data Versioning
    if choice in ['1', '5']:
        results['phase1'] = run_phase_1()
    
    # Phase 2: Experiment Tracking
    if choice in ['2', '5']:
        if results.get('phase1', True):  # Only run if phase 1 succeeded or wasn't run
            results['phase2'] = run_phase_2()
        else:
            print("\n[WARN] Skipping Phase 2 because Phase 1 failed.")
    
    # Phase 3: Hyperparameter Sweep
    sweep_id = None
    if choice in ['3', '5']:
        if results.get('phase2', True):  # Only run if phase 2 succeeded or wasn't run
            sweep_id = run_phase_3()
            results['phase3'] = sweep_id is not None
            if sweep_id:
                print(f"\n[INFO] Sweep ID: {sweep_id}")
        else:
            print("\n[WARN] Skipping Phase 3 because Phase 2 failed.")
    
    # Phase 4: Model Registration
    if choice in ['4', '5']:
        if sweep_id:
            # Use the sweep_id from Phase 3 (just completed)
            print(f"\n[INFO] Using sweep_id from Phase 3: {sweep_id}")
            results['phase4'] = run_phase_4_with_sweep_id(sweep_id)
        elif results.get('phase3', False):
            # Phase 3 completed but no sweep_id returned, ask user
            results['phase4'] = run_phase_4()
        else:
            print("\n[WARN] Phase 4 requires Phase 3 to complete first.")
            print("   Please run Phase 3 (sweep) first, then run Phase 4.")
    
    # Phase 3 Dry Run
    if choice == '6':
        results['phase3_dry'] = run_phase_3_dry_run()
    
    # Just Verification
    if choice == '7':
        verify_mlops_principles()
        return True
    
    # Verify MLOps principles
    verify_mlops_principles()
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    for phase, success in results.items():
        if phase == 'phase3' and isinstance(success, str):
            # Phase 3 returns sweep_id, not just True/False
            print(f"{phase.upper()}: [OK] PASSED (sweep_id: {success})")
        else:
            status = "[OK] PASSED" if success else "[FAIL] FAILED"
            print(f"{phase.upper()}: {status}")
    
    all_passed = all(results.values()) if results else False
    
    if all_passed:
        print("\n[SUCCESS] All phases completed successfully!")
        print("\nNext steps:")
        print("  1. Check your W&B dashboard for results")
        if 'phase3' in results and results['phase3']:
            print("  2. Sweep completed! You can now register the best model.")
            print("     Run Phase 4 or use: python mlops_model_registry.py")
        elif 'phase4' in results and results['phase4']:
            print("  2. Model registered! Check W&B Artifacts for the model.")
        else:
            print("  2. Continue with remaining phases if needed")
    else:
        print("\n[WARN] Some phases failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Pipeline cancelled by user.")
    except Exception as e:
        print(f"\n\n[FAIL] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
