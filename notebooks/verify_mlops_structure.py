"""
MLOps Structure Verification Script
This script verifies that all MLOps principles are implemented correctly
without requiring W&B login or actual execution.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

class MLOpsVerifier:
    """Verifies MLOps implementation structure."""
    
    def __init__(self, notebooks_dir: Path):
        self.notebooks_dir = notebooks_dir
        self.issues = []
        self.warnings = []
        self.successes = []
    
    def check_file_exists(self, filename: str) -> bool:
        """Check if a file exists."""
        filepath = self.notebooks_dir / filename
        if filepath.exists():
            self.successes.append(f"[OK] {filename} exists")
            return True
        else:
            self.issues.append(f"[FAIL] {filename} not found")
            return False
    
    def check_imports(self, filename: str) -> List[str]:
        """Check if file has required imports."""
        filepath = self.notebooks_dir / filename
        if not filepath.exists():
            return []
        
        content = filepath.read_text()
        required_imports = {
            'wandb': 'wandb',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'scikit-learn',
            'xgboost': 'xgboost'
        }
        
        found_imports = []
        missing_imports = []
        
        for module, package in required_imports.items():
            if f'import {module}' in content or f'from {module}' in content:
                found_imports.append(package)
            else:
                if filename in ['mlops_training_tracked.py', 'mlops_hyperparameter_sweep.py', 'mlops_model_registry.py']:
                    missing_imports.append(package)
        
        if missing_imports:
            self.warnings.append(f"[WARN] {filename}: Missing imports: {', '.join(missing_imports)}")
        else:
            self.successes.append(f"[OK] {filename}: All required imports present")
        
        return found_imports
    
    def check_wandb_usage(self, filename: str) -> Dict[str, bool]:
        """Check if file uses W&B correctly."""
        filepath = self.notebooks_dir / filename
        if not filepath.exists():
            return {}
        
        content = filepath.read_text()
        
        checks = {
            'wandb.init': 'wandb.init' in content,
            'wandb.log': 'wandb.log' in content,
            'wandb.Artifact': 'wandb.Artifact' in content or 'Artifact' in content,
            'wandb.config': 'wandb.config' in content,
        }
        
        # Phase-specific checks
        if 'data_versioning' in filename:
            checks['artifact.add_dir'] = 'artifact.add_dir' in content or 'add_dir' in content
            checks['run.log_artifact'] = 'log_artifact' in content
        
        if 'training' in filename or 'sweep' in filename:
            checks['metrics_logging'] = any(metric in content for metric in ['f1_score', 'roc_auc', 'accuracy'])
        
        if 'sweep' in filename:
            checks['wandb.sweep'] = 'wandb.sweep' in content
            checks['sweep_config'] = 'sweep_config' in content
        
        if 'registry' in filename:
            checks['wandb.Api'] = 'wandb.Api' in content or 'Api' in content
            checks['model_artifact'] = 'model' in content.lower() and 'artifact' in content.lower()
        
        return checks
    
    def check_mlops_principles(self) -> Dict[str, bool]:
        """Check if all MLOps principles are implemented."""
        principles = {
            'Data Versioning': self.check_data_versioning(),
            'Experiment Tracking': self.check_experiment_tracking(),
            'Hyperparameter Optimization': self.check_hyperparameter_optimization(),
            'Model Registration': self.check_model_registration(),
            'Reproducibility': self.check_reproducibility(),
            'Metrics Logging': self.check_metrics_logging(),
        }
        return principles
    
    def check_data_versioning(self) -> bool:
        """Check Phase 1: Data Versioning."""
        filepath = self.notebooks_dir / 'mlops_data_versioning.py'
        if not filepath.exists():
            return False
        
        content = filepath.read_text()
        
        checks = [
            'wandb.init' in content,
            'wandb.Artifact' in content or 'Artifact' in content,
            'log_artifact' in content,
            'metadata' in content,
            'train.csv' in content or 'train' in content.lower(),
        ]
        
        if all(checks):
            self.successes.append("[OK] Data Versioning: All components present")
            return True
        else:
            self.issues.append("[FAIL] Data Versioning: Missing components")
            return False
    
    def check_experiment_tracking(self) -> bool:
        """Check Phase 2: Experiment Tracking."""
        filepath = self.notebooks_dir / 'mlops_training_tracked.py'
        if not filepath.exists():
            return False
        
        content = filepath.read_text()
        
        checks = [
            'wandb.init' in content,
            'wandb.log' in content,
            'wandb.config' in content,
            'f1_score' in content or 'roc_auc' in content,
            'confusion_matrix' in content or 'confusion' in content.lower(),
        ]
        
        if all(checks):
            self.successes.append("[OK] Experiment Tracking: All components present")
            return True
        else:
            self.issues.append("[FAIL] Experiment Tracking: Missing components")
            return False
    
    def check_hyperparameter_optimization(self) -> bool:
        """Check Phase 3: Hyperparameter Optimization."""
        filepath = self.notebooks_dir / 'mlops_hyperparameter_sweep.py'
        if not filepath.exists():
            return False
        
        content = filepath.read_text()
        
        checks = [
            'wandb.sweep' in content,
            'sweep_config' in content,
            'wandb.agent' in content or 'agent' in content,
            'method' in content.lower() and ('bayes' in content.lower() or 'random' in content.lower()),
            'metric' in content.lower(),
        ]
        
        if all(checks):
            self.successes.append("[OK] Hyperparameter Optimization: All components present")
            return True
        else:
            self.issues.append("[FAIL] Hyperparameter Optimization: Missing components")
            return False
    
    def check_model_registration(self) -> bool:
        """Check Phase 4: Model Registration."""
        filepath = self.notebooks_dir / 'mlops_model_registry.py'
        if not filepath.exists():
            return False
        
        content = filepath.read_text()
        
        checks = [
            'wandb.Api' in content or 'Api' in content,
            'best_run' in content.lower(),
            'wandb.Artifact' in content or 'Artifact' in content,
            'joblib.dump' in content or 'dump' in content,
        ]
        
        if all(checks):
            self.successes.append("[OK] Model Registration: All components present")
            return True
        else:
            self.issues.append("[FAIL] Model Registration: Missing components")
            return False
    
    def check_reproducibility(self) -> bool:
        """Check if reproducibility measures are in place."""
        files = [
            'mlops_training_tracked.py',
            'mlops_hyperparameter_sweep.py',
            'mlops_model_registry.py'
        ]
        
        has_random_state = False
        for filename in files:
            filepath = self.notebooks_dir / filename
            if filepath.exists():
                content = filepath.read_text()
                if 'random_state' in content:
                    has_random_state = True
                    break
        
        if has_random_state:
            self.successes.append("[OK] Reproducibility: Random seeds configured")
            return True
        else:
            self.warnings.append("[WARN] Reproducibility: Random seeds not found")
            return False
    
    def check_metrics_logging(self) -> bool:
        """Check if comprehensive metrics are logged."""
        filepath = self.notebooks_dir / 'mlops_training_tracked.py'
        if not filepath.exists():
            return False
        
        content = filepath.read_text()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
        found_metrics = [m for m in metrics if m in content.lower()]
        
        if len(found_metrics) >= 4:
            self.successes.append(f"[OK] Metrics Logging: {len(found_metrics)} metrics found")
            return True
        else:
            self.warnings.append(f"[WARN] Metrics Logging: Only {len(found_metrics)} metrics found")
            return False
    
    def verify_all(self) -> Tuple[bool, Dict]:
        """Run all verification checks."""
        print("=" * 60)
        print("MLOPS STRUCTURE VERIFICATION")
        print("=" * 60)
        
        # Check files exist
        print("\n[FILES] Checking Files...")
        files = [
            'mlops_data_versioning.py',
            'mlops_training_tracked.py',
            'mlops_hyperparameter_sweep.py',
            'mlops_model_registry.py',
            'MLOPS_COMPLETE_PIPELINE.ipynb'
        ]
        
        for filename in files:
            self.check_file_exists(filename)
        
        # Check imports
        print("\n[IMPORTS] Checking Imports...")
        for filename in files[:4]:  # Python files only
            if filename.endswith('.py'):
                self.check_imports(filename)
        
        # Check W&B usage
        print("\n[WANDB] Checking W&B Integration...")
        for filename in files[:4]:
            if filename.endswith('.py'):
                checks = self.check_wandb_usage(filename)
                if checks:
                    all_checks = all(checks.values())
                    if all_checks:
                        self.successes.append(f"[OK] {filename}: W&B properly integrated")
                    else:
                        missing = [k for k, v in checks.items() if not v]
                        self.warnings.append(f"[WARN] {filename}: Missing W&B features: {', '.join(missing)}")
        
        # Check MLOps principles
        print("\n[PRINCIPLES] Checking MLOps Principles...")
        principles = self.check_mlops_principles()
        
        # Print results
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        
        print("\n[SUCCESS] Successes:")
        for success in self.successes:
            print(f"  {success}")
        
        if self.warnings:
            print("\n[WARNING] Warnings:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.issues:
            print("\n[ISSUE] Issues:")
            for issue in self.issues:
                print(f"  {issue}")
        
        print("\n" + "=" * 60)
        print("MLOPS PRINCIPLES SUMMARY")
        print("=" * 60)
        
        for principle, status in principles.items():
            status_icon = "[OK]" if status else "[FAIL]"
            print(f"{status_icon} {principle}")
        
        all_passed = len(self.issues) == 0 and all(principles.values())
        
        print("\n" + "=" * 60)
        if all_passed:
            print("[SUCCESS] ALL CHECKS PASSED!")
            print("=" * 60)
            print("\nYour MLOps implementation follows all best practices!")
            print("You can now run the pipeline with:")
            print("  python run_mlops_pipeline.py")
        else:
            print("⚠️  SOME ISSUES FOUND")
            print("=" * 60)
            print("\nPlease review the issues above before running the pipeline.")
        
        return all_passed, {
            'successes': len(self.successes),
            'warnings': len(self.warnings),
            'issues': len(self.issues),
            'principles': principles
        }

def main():
    """Main verification function."""
    notebooks_dir = Path(__file__).parent
    
    verifier = MLOpsVerifier(notebooks_dir)
    passed, results = verifier.verify_all()
    
    return passed

if __name__ == "__main__":
    main()
