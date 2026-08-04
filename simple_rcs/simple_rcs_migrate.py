import os
import sys
from typing import List, Callable, Tuple, Optional
from simple_rcs.simple_rcs import SimpleRCS

def migrate(
    src_path: str, 
    dst_path: str, 
    signer_callbacks: Optional[List[Callable[[str], Tuple[str, str]]]] = None
) -> bool:
    """
    Migrates a SimpleRCS v1 file to v2 format.
    
    This process reads the entire history from the source file (v1) and
    re-commits each version sequentially into a new destination file (v2).
    This establishes the hash chain and allows applying signatures to the entire history.
    
    Args:
        src_path: Path to the existing v1 file.
        dst_path: Path for the new v2 file. Should not exist.
        signer_callbacks: Optional list of signing functions to sign each version during migration.
        
    Returns:
        True if successful, False otherwise.
    """
    
    if not os.path.exists(src_path):
        print(f"Error: Source file '{src_path}' not found.")
        return False
        
    if os.path.exists(dst_path):
        print(f"Error: Destination file '{dst_path}' already exists.")
        return False
        
    try:
        # 1. Open Source (v1)
        src_rcs = SimpleRCS(src_path)
        
        # 2. Open Destination (v2) - New file creation defaults to v2
        dst_rcs = SimpleRCS(dst_path)
        
        # 3. Get Full History (Oldest to Newest)
        history = src_rcs.log(reverse=True)
        if not history:
            print("Source file has no history.")
            return True
            
        print(f"Migrating {len(history)} revisions from '{src_path}' to '{dst_path}'...")
        
        for entry in history:
            ver = entry['ver']
            print(f"  Processing version {ver}...")
            
            # Checkout Full Text from v1
            content = src_rcs.checkout(ver)
            
            # Commit to v2
            # We preserve author, log, and date.
            # Signer callbacks (if any) will sign this new block.
            dst_rcs.commit(
                content=content,
                author=entry['author'],
                log=entry['log'],
                date=entry['date'],
                signer_callbacks=signer_callbacks
            )
            
        print("Migration complete.")
        
        # Optional: Verify the new file
        if dst_rcs.verify():
            print("Verification successful: New file integrity checks pass.")
            return True
        else:
            print("Warning: Verification failed on the new file.")
            return False
            
    except Exception as e:
        print(f"Migration failed: {e}")
        # Cleanup partial file
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 simple_rcs_migrate.py <src_v1_path> <dst_v2_path>")
        sys.exit(1)
        
    src = sys.argv[1]
    dst = sys.argv[2]
    
    success = migrate(src, dst)
    sys.exit(0 if success else 1)
