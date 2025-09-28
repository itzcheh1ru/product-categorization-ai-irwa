#!/usr/bin/env python3
"""
Data Migration Script: CSV to MongoDB
Migrates product data from CSV files to MongoDB Atlas
"""

import sys
import os
from pathlib import Path
import logging

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.mongodb_connection import MongoDBManager, connect_to_mongodb, migrate_csv_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main migration function"""
    print("🚀 Starting CSV to MongoDB Migration")
    print("=" * 50)
    
    # Check if CSV file exists
    csv_file_path = "data/cleaned_product_data.csv"
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        print("Available files in data directory:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.iterdir():
                print(f"  - {file.name}")
        return False
    
    print(f"📁 Found CSV file: {csv_file_path}")
    
    # Connect to MongoDB
    print("\n🔌 Connecting to MongoDB Atlas...")
    if not connect_to_mongodb():
        print("❌ Failed to connect to MongoDB Atlas")
        return False
    
    print("✅ Successfully connected to MongoDB Atlas")
    
    # Get database stats before migration
    print("\n📊 Database Status Before Migration:")
    mongodb_manager = MongoDBManager()
    stats = mongodb_manager.get_database_stats()
    print(f"  - Total Products: {stats.get('total_products', 0)}")
    print(f"  - Categories: {stats.get('categories', 0)}")
    print(f"  - Brands: {stats.get('brands', 0)}")
    
    # Migrate data
    print(f"\n📤 Migrating data from {csv_file_path} to MongoDB...")
    if migrate_csv_data(csv_file_path):
        print("✅ Data migration completed successfully!")
        
        # Get database stats after migration
        print("\n📊 Database Status After Migration:")
        stats = mongodb_manager.get_database_stats()
        print(f"  - Total Products: {stats.get('total_products', 0)}")
        print(f"  - Categories: {stats.get('categories', 0)}")
        print(f"  - Brands: {stats.get('brands', 0)}")
        
        # Create text index for better search performance
        print("\n🔍 Creating search indexes...")
        mongodb_manager.create_text_index()
        print("✅ Search indexes created successfully!")
        
        # Test search functionality
        print("\n🧪 Testing search functionality...")
        test_products = mongodb_manager.search_products("shirt", limit=3)
        print(f"  - Found {len(test_products)} products matching 'shirt'")
        
        if test_products:
            print("  - Sample product:")
            sample = test_products[0]
            print(f"    Name: {sample.get('name', 'N/A')}")
            print(f"    Category: {sample.get('category', 'N/A')}")
            print(f"    Brand: {sample.get('brand', 'N/A')}")
        
        print("\n🎉 Migration completed successfully!")
        print("✅ MongoDB is ready for use")
        return True
        
    else:
        print("❌ Data migration failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Next steps:")
        print("1. Update your API endpoints to use MongoDB")
        print("2. Test the new MongoDB integration")
        print("3. Remove CSV file dependencies")
        sys.exit(0)
    else:
        print("\n❌ Migration failed. Please check the logs and try again.")
        sys.exit(1)
