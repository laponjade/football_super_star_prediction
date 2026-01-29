"""
Quick script to check if model is registered in W&B
Run this to verify if your model has been registered.
"""

from wandb import Api
from mlops_config import ENTITY, PROJECT

def check_model_registration():
    """Check if the model artifact exists in W&B."""
    api = Api()
    
    artifact_name = "football-superstar-predictor"
    
    try:
        # Try to get the model artifact
        artifact = api.artifact(f"{ENTITY}/{PROJECT}/{artifact_name}:latest")
        
        print("=" * 60)
        print("[SUCCESS] MODEL IS REGISTERED!")
        print("=" * 60)
        print(f"Artifact Name: {artifact.name}")
        print(f"Version: {artifact.version}")
        print(f"Type: {artifact.type}")
        print(f"Created: {artifact.created_at}")
        
        print(f"\n[FILES] Files in Artifact:")
        for file in artifact.files:
            size_kb = file.size / 1024
            print(f"  - {file.name} ({size_kb:.2f} KB)")
        
        print(f"\n[METADATA] Metadata:")
        if artifact.metadata:
            for key, value in artifact.metadata.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value}")
                elif isinstance(value, dict):
                    print(f"  - {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"  - {key}: {value}")
        else:
            print("  (No metadata)")
        
        print(f"\n[URL] View at: {artifact.url}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print("[INFO] MODEL IS NOT REGISTERED YET")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("\n[INFO] To register a model:")
        print("  1. Run Phase 3: python mlops_hyperparameter_sweep.py")
        print("  2. Wait for sweep to complete")
        print("  3. Get sweep_id from W&B dashboard")
        print("  4. Run: python mlops_model_registry.py")
        print("     Or use: register_best_model(sweep_id='your-sweep-id')")
        print("=" * 60)
        
        return False

if __name__ == "__main__":
    check_model_registration()
