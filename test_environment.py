#!/usr/bin/env python3
"""
Test script to verify that the WAM environment is properly set up.
Run this script after setting up the environment to ensure everything works.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported successfully."""
    try:
        if package_name:
            module = importlib.import_module(module_name, package=package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Warning importing {module_name}: {e}")
        return True

def test_torch_cuda():
    """Test if PyTorch can access CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA is not available (CPU-only mode)")
            return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def test_wam_imports():
    """Test if WAM modules can be imported."""
    try:
        # Test lib imports
        from lib import wam_2D
        print("✅ WAM 2D module imported successfully")
        
        # Test src imports
        from src import helpers, dataloader, network_architectures
        print("✅ WAM source modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import WAM modules: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Warning importing WAM modules: {e}")
        return True

def main():
    """Run all environment tests."""
    print("🧪 Testing WAM Environment Setup")
    print("=" * 40)
    
    # Test Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major == 3 and python_version.minor >= 8:
        print("✅ Python version is compatible")
    else:
        print("❌ Python version should be 3.8 or higher")
        return False
    
    print("\n📦 Testing Core Dependencies:")
    print("-" * 30)
    
    # Test core dependencies
    core_modules = [
        "torch",
        "torchvision", 
        "numpy",
        "matplotlib",
        "cv2",
        "ptwt",
        "timm",
        "scipy",
        "pandas",
        "PIL"
    ]
    
    all_core_ok = True
    for module in core_modules:
        if not test_import(module):
            all_core_ok = False
    
    print("\n🎮 Testing CUDA Support:")
    print("-" * 30)
    cuda_ok = test_torch_cuda()
    
    print("\n🔬 Testing WAM Modules:")
    print("-" * 30)
    wam_ok = test_wam_imports()
    
    print("\n" + "=" * 40)
    if all_core_ok and wam_ok:
        print("🎉 All tests passed! Your WAM environment is ready to use.")
        print("\nNext steps:")
        print("1. Activate your environment: conda activate wam")
        print("2. Open the example notebook: jupyter notebook example.ipynb")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct conda environment")
        print("2. Try reinstalling dependencies: pip install -r requirements.txt")
        print("3. Check the README.md for troubleshooting tips")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 