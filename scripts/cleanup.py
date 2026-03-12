
import os
import shutil
import glob
import argparse

def cleanup(dry_run=True):
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
    
    if not dry_run:
        confirm = input("Are you absolutely sure you want to delete these files? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return

    print("Cleaning up...")
    
    for f in files_to_delete:
        if os.path.exists(f):
            if not dry_run:
                os.remove(f)
            print(f"{'Would delete' if dry_run else 'Deleted'} {f}")
            
    for d in dirs_to_delete:
        if os.path.exists(d):
            if not dry_run:
                shutil.rmtree(d)
            print(f"{'Would delete directory' if dry_run else 'Deleted directory'} {d}")
            
    # Clean up any pyc files recursively
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc"):
                path = os.path.join(root, file)
                if not dry_run:
                    os.remove(path)
                print(f"{'Would delete' if dry_run else 'Deleted'} {path}")
                
    if dry_run:
        print("\nDry run complete. Use --no-dry-run to actually delete files.")
    else:
        print("Cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up repository artifacts.")
    parser.add_argument('--no-dry-run', action='store_true', help="Actually perform the deletions.")
    args = parser.parse_args()
    cleanup(dry_run=not args.no_dry_run)
