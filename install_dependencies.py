#!/usr/bin/env python3
"""
Script to install dependencies for the fine-tuning system
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install requirements from requirements.txt"""
    try:
        # Get the backend directory
        backend_dir = Path(__file__).parent / "backend"
        requirements_file = backend_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print(f"❌ Requirements file not found at {requirements_file}")
            return False
        
        print(f"📦 Installing dependencies from {requirements_file}")
        print("This may take a few minutes...")
        
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            print("📋 Installed packages:")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to install dependencies!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_installation():
    """Check if key packages are installed"""
    required_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "evaluate", "scikit-learn", "matplotlib", "seaborn"
    ]
    
    print("\n🔍 Checking installation...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - not installed")
            return False
    
    return True

def main():
    print("🚀 Installing Fine-Tuning Dependencies")
    print("=" * 50)
    
    # Install requirements
    if install_requirements():
        # Check installation
        if check_installation():
            print("\n🎉 All dependencies installed successfully!")
            print("You can now run the fine-tuning system.")
            return 0
        else:
            print("\n⚠️ Some packages may not be installed correctly.")
            return 1
    else:
        print("\n💥 Failed to install dependencies!")
        return 1

if __name__ == "__main__":
    exit(main())
