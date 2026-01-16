#!/usr/bin/env python3
"""
Master script to run the three-class pipeline in sequence:
1. three_class_test.py - Three-class classification (cassette vs alt3 vs alt5)
2. binary_class_test.py - Binary classification (cassette vs alt3, cassette vs alt5, alt3 vs alt5)
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_name, description, args=None):
    """Run a Python script and return success status"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__),
            check=True,
            capture_output=False,
            text=True
        )
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {description} failed with error: {str(e)}")
        return False

def main():
    """Main function to run all tests in sequence"""
    if len(sys.argv) > 1:
        dataset_arg = sys.argv[1]
    else:
        dataset_arg = "dataset1"
        print("Usage: python run_all_tests.py [dataset1|dataset2|dataset3|<custom_path>]")
    
    # Define all paths centrally here
    dataset_names = {
        "dataset1": "dataset1_constitutive_cassette_dominant.csv",
        "dataset2": "dataset2_alt_three_alt_five_dominant.csv",
        "dataset3": "dataset3_constitutive_alt_three_dominant.csv",
        "balanced_multiclass_dataset": "../data_preprocessing/balanced_multiclass_data/test.csv"
    }
    
    # Generate timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if it's a predefined dataset or a custom path
    if dataset_arg in dataset_names:
        # Predefined dataset
        dataset = dataset_arg
        test_csv_path = dataset_names[dataset]
        # Create unique folder with timestamp
        base_output_dir = f"./test_result_{dataset}_{timestamp}"
    else:
        dataset = os.path.basename(dataset_arg).replace('.csv', '').replace('/', '_').replace('\\', '_')
        test_csv_path = dataset_arg
        base_output_dir = f"./test_result_{dataset}_{timestamp}"
    
    three_class_output_dir = os.path.join(base_output_dir, "result_three_class")
    binary_class_output_dir = os.path.join(base_output_dir, "result_binary_class")
    three_class_output_file = os.path.join(three_class_output_dir, "predictions_with_probabilities.csv")
    
    # Step 1: Run three-class classification
    three_class_args = ["--output-dir", three_class_output_dir]
    if dataset_arg in dataset_names:
        if dataset_arg == "balanced_multiclass_dataset":
            three_class_args.extend(["--input-csv", test_csv_path])
        else:
            three_class_args.extend(["--dataset", dataset])
    else:
        three_class_args.extend(["--input-csv", test_csv_path])
    
    success = run_script(
        "three_class_test.py",
        "Three-Class Classification",
        args=three_class_args
    )
    
    if not success:
        print("ERROR: Three-class classification failed. Stopping pipeline.")
        sys.exit(1)
    
    if not os.path.exists(three_class_output_file):
        print(f"ERROR: Expected output not found: {three_class_output_file}")
        sys.exit(1)
    
    # Step 2: Run binary classification
    success = run_script(
        "binary_class_test.py",
        "Binary Classification",
        args=["--input", three_class_output_file, "--output-dir", binary_class_output_dir]
    )
    
    if not success:
        print("ERROR: Binary classification failed. Stopping pipeline.")
        sys.exit(1)
    
    print(f"\nPipeline completed successfully. Results saved in: {base_output_dir}/")

if __name__ == "__main__":
    main()

