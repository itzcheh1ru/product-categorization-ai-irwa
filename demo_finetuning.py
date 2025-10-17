#!/usr/bin/env python3
"""
Demo script showing the fine-tuning system working
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def demo_data_preparation():
    """Demonstrate data preparation"""
    print("🔍 Demo: Data Preparation")
    print("-" * 40)
    
    try:
        from backend.data_finetuned.training_data_preparation import TrainingDataPreparation
        
        # Initialize data preparation
        data_prep = TrainingDataPreparation()
        
        # Load and show data statistics
        df = data_prep.load_product_data()
        print(f"✅ Loaded {len(df)} products")
        print(f"📊 Categories: {df['masterCategory'].nunique()}")
        print(f"📊 Subcategories: {df['subCategory'].nunique()}")
        print(f"📊 Article types: {df['articleType'].nunique()}")
        
        # Show sample data
        print("\n📋 Sample products:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. {row['productDisplayName']} - {row['masterCategory']} - {row['subCategory']}")
        
        # Prepare training data
        print("\n🔄 Preparing training data...")
        training_data = data_prep.prepare_all_training_data()
        
        print(f"✅ Training data prepared!")
        print(f"   Classification: {len(training_data['classification']['train'])} train, {len(training_data['classification']['validation'])} validation")
        
        # Show sample training example
        sample = training_data['classification']['train'][0]
        print(f"\n📝 Sample training example:")
        print(f"   Input: {sample['input_text'][:100]}...")
        print(f"   Labels: {sample['labels']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_fine_tuning_setup():
    """Demonstrate fine-tuning setup"""
    print("\n🔍 Demo: Fine-Tuning Setup")
    print("-" * 40)
    
    try:
        from backend.data_finetuned.fine_tuning import FineTuningManager
        
        # Initialize fine-tuning manager
        manager = FineTuningManager()
        print("✅ Fine-tuning manager initialized")
        
        # Show configuration
        from backend.data_finetuned.training_data_preparation import TrainingDataPreparation
        data_prep = TrainingDataPreparation()
        config = data_prep.create_training_config()
        
        print(f"📋 Training configuration:")
        print(f"   Model: {config['model_name']}")
        print(f"   Epochs: {config['num_epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning rate: {config['learning_rate']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_model_integration():
    """Demonstrate model integration"""
    print("\n🔍 Demo: Model Integration")
    print("-" * 40)
    
    try:
        from backend.data_finetuned.fine_tuned_classifier import fine_tuned_classifier
        
        # Check model availability
        if fine_tuned_classifier.is_model_available():
            print("✅ Fine-tuned model is available!")
            
            # Test prediction
            test_product = {
                'productDisplayName': 'Nike Men Blue T-shirt',
                'description': 'Men blue cotton t-shirt for casual wear'
            }
            
            result = fine_tuned_classifier.classify_product(test_product)
            print(f"🎯 Test prediction: {result}")
            
        else:
            print("ℹ️  Fine-tuned model not available (need to train first)")
            print("   This is expected - you need to run training first")
            
            # Show fallback
            print("🔄 Using fallback classification...")
            result = fine_tuned_classifier._fallback_classification(test_product)
            print(f"🎯 Fallback prediction: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_training_commands():
    """Show training commands"""
    print("\n🔍 Demo: Training Commands")
    print("-" * 40)
    
    print("📋 To train your fine-tuned models, run:")
    print("   python backend/scripts/train_and_evaluate.py")
    print()
    print("📋 To evaluate models:")
    print("   python backend/scripts/train_and_evaluate.py --evaluate-only")
    print()
    print("📋 To compare models:")
    print("   python backend/scripts/train_and_evaluate.py --evaluate-only --compare-models")
    print()
    print("📋 To test single prediction:")
    print('   python backend/scripts/train_and_evaluate.py --test-prediction "Nike Men Blue T-shirt|Men blue cotton t-shirt"')
    
    return True

def main():
    """Run demo"""
    print("🚀 Fine-Tuning System Demo")
    print("=" * 50)
    
    demos = [
        ("Data Preparation", demo_data_preparation),
        ("Fine-Tuning Setup", demo_fine_tuning_setup),
        ("Model Integration", demo_model_integration),
        ("Training Commands", demo_training_commands)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        success = demo_func()
        results.append((demo_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{demo_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("\n🎉 Fine-tuning system is ready!")
        print("📋 Next steps:")
        print("   1. Run training: python backend/scripts/train_and_evaluate.py")
        print("   2. Test your fine-tuned models")
        print("   3. Enjoy improved categorization accuracy!")
        return 0
    else:
        print("\n⚠️  Some demos failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
