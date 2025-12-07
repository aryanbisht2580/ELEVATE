import os
import sys
import torch
from resume_enhancement_train import train_resume_enhancer, load_and_test_model

def check_dataset():
    """Check if para-nmt dataset exists and is accessible"""
    dataset_path = 'para-nmt-50m.txt'
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found: {dataset_path}")
        
        # Check first few lines to confirm format
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Just check first 3 lines
                    break
                print(f"  Sample line {i+1}: {line.strip()[:100]}...")
        return True
    else:
        print(f"✗ Dataset not found: {dataset_path}")
        return False

def main():
    print("Resume Enhancement Model Setup")
    print("="*40)
    
    # Check if dataset exists
    if not check_dataset():
        print("\nError: para-nmt-50m.txt not found in the ML model directory.")
        print("Please make sure the file exists before running the training.")
        return
    
    # Check arguments to decide what to do
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            print("\nStarting training process...")
            train_resume_enhancer()
        elif sys.argv[1] == 'test':
            print("\nLoading and testing existing model...")
            load_and_test_model()
        else:
            print(f"\nUnknown command: {sys.argv[1]}")
            print("Usage: python setup_resume_model.py [train|test]")
    else:
        print("\nDefaulting to training mode...")
        print("Usage: python setup_resume_model.py [train|test]")
        train_resume_enhancer()

if __name__ == '__main__':
    main()