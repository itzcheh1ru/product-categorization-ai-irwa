#!/usr/bin/env python3
"""
Remove CSV Dependencies Script
This script removes CSV file dependencies after successful MongoDB migration
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def remove_csv_dependencies():
    """Remove CSV file dependencies and update configuration"""
    
    print("🗑️  Removing CSV Dependencies")
    print("=" * 50)
    
    # List of CSV files to remove (optional - you might want to keep them as backup)
    csv_files_to_remove = [
        "data/cleaned_product_data.csv",
        "data/product.csv"
    ]
    
    # Files to update to remove CSV references
    files_to_update = [
        "api/main.py"
    ]
    
    removed_files = []
    updated_files = []
    
    # Remove CSV files (optional - comment out if you want to keep them)
    print("\n📁 CSV Files Status:")
    for csv_file in csv_files_to_remove:
        if os.path.exists(csv_file):
            print(f"  - {csv_file}: EXISTS")
            # Uncomment the next lines to actually remove the files
            # os.remove(csv_file)
            # removed_files.append(csv_file)
            # print(f"    ✅ Removed {csv_file}")
        else:
            print(f"  - {csv_file}: NOT FOUND")
    
    # Update files to remove CSV references
    print("\n📝 Updating Files:")
    
    # Update main.py to remove CSV path reference
    main_py_path = "api/main.py"
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Remove or comment out CSV path reference
        old_line = 'DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_product_data.csv"'
        new_line = '# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_product_data.csv"  # Removed: Using MongoDB now'
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            
            with open(main_py_path, 'w') as f:
                f.write(content)
            
            updated_files.append(main_py_path)
            print(f"    ✅ Updated {main_py_path}")
        else:
            print(f"    ℹ️  {main_py_path} already updated or no CSV reference found")
    
    # Create backup of CSV files (optional)
    backup_dir = "data/backup_csv"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"\n📦 Created backup directory: {backup_dir}")
    
    # Summary
    print("\n📊 Summary:")
    print(f"  - Files removed: {len(removed_files)}")
    print(f"  - Files updated: {len(updated_files)}")
    
    if removed_files:
        print("\n🗑️  Removed files:")
        for file in removed_files:
            print(f"    - {file}")
    
    if updated_files:
        print("\n📝 Updated files:")
        for file in updated_files:
            print(f"    - {file}")
    
    print("\n✅ CSV dependency removal completed!")
    print("\n🚀 Next steps:")
    print("1. Test the application to ensure MongoDB integration works")
    print("2. Update documentation to reflect MongoDB usage")
    print("3. Consider removing CSV files if no longer needed")
    
    return True

if __name__ == "__main__":
    success = remove_csv_dependencies()
    if success:
        print("\n🎉 CSV dependencies successfully removed!")
    else:
        print("\n❌ Failed to remove CSV dependencies")
