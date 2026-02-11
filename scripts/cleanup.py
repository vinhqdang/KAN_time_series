
import os
import shutil
import glob

def cleanup():
    # Files to delete
    files_to_delete = [
        "benchmark_predictive.py", # Old script
        "validate_adakan.py",      # Old script
        "cdkan_adjacency.png",     # Artifact (moved to docs/artifacts if needed, or just keep result)
        # We keep csv results
    ]
    
    # Directors to delete
    dirs_to_delete = [
        "__pycache__",
        "src/__pycache__",
        "src/cdkan/__pycache__",
        "src/adakan/__pycache__",
        ".pytest_cache"
    ]
    
    print("Cleaning up...")
    
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")
            
    for d in dirs_to_delete:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Deleted directory {d}")
            
    # Clean up any pyc files recursively
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc"):
                os.remove(os.path.join(root, file))
                
    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup()
