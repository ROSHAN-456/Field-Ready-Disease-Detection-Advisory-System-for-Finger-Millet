"""
run_pipeline.py — Master orchestrator to run the entire data + training pipeline.

1. Swapping to real data:
   Just point this script to your real raw folder and let it do everything else.
   
Usage:
   python run_pipeline.py --raw_data "d:\ML pj\real_raw_dataset"
"""
import os
import sys
import argparse
import subprocess
import shutil

# Ensure we can import from dataset_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset_pipeline'))

def run_cmd(cmd, cwd=None):
    print(f"\n[EXEC] Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Finger Millet Pipeline")
    parser.add_argument("--raw_data", type=str, 
                        help="Path to raw dataset folder (leave empty to use synthetic)")
    parser.add_argument("--aug_target", type=int, default=500,
                        help="Target number of images per class after augmentation (default: 500)")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_code_dir = os.path.abspath(os.path.join(base_dir, "..", "dataset_pipeline"))
    training_code_dir = base_dir
    
    # Unified output folders
    augmented_dir = os.path.abspath(os.path.join(dataset_code_dir, "dataset_augmented"))
    split_dir = os.path.abspath(os.path.join(dataset_code_dir, "dataset_split"))
    reports_dir = os.path.abspath(os.path.join(training_code_dir, "eval_results"))
    benchmark_dir = os.path.abspath(os.path.join(training_code_dir, "benchmark_data"))

    if args.raw_data:
        raw_dir = os.path.abspath(args.raw_data)
        if not os.path.exists(raw_dir):
            print(f"[ERROR] Target raw data directory not found: {raw_dir}")
            sys.exit(1)
            
        print("\n" + "="*60)
        print(" 1. DATA PREPARATION (VALIDATE -> AUGMENT -> SPLIT)")
        print("="*60)
        
        # We will write a tiny wrapper python call to avoid the `input()` prompts in the original scripts
        
        # 1a Validator
        print("\n[STEP 1A] Validating Schema")
        run_cmd(f"python -c \"import validate_dataset; validate_dataset.validate_dataset('{raw_dir.replace(chr(92), '/')}')\"", cwd=dataset_code_dir)
        
        # 1b Augmentor (Clear old first)
        if os.path.exists(augmented_dir): shutil.rmtree(augmented_dir)
        print("\n[STEP 1B] Augmenting Data to simulate field conditions")
        run_cmd(f"python -c \"import augment_dataset; augment_dataset.run_augmentation('{raw_dir.replace(chr(92), '/')}', '{augmented_dir.replace(chr(92), '/')}', {args.aug_target})\"", cwd=dataset_code_dir)
        
        # 1c Splitter
        if os.path.exists(split_dir): shutil.rmtree(split_dir)
        print("\n[STEP 1C] Stratifying Train/Val/Test Splits")
        run_cmd(f"python -c \"import split_dataset; split_dataset.create_stratified_split('{augmented_dir.replace(chr(92), '/')}', '{split_dir.replace(chr(92), '/')}')\"", cwd=dataset_code_dir)
        
        # 1d Quality Report
        os.makedirs(reports_dir, exist_ok=True)
        report_out = os.path.join(reports_dir, "dataset_quality.md")
        print("\n[STEP 1D] Generating Dataset Markdown Report")
        run_cmd(f"python -c \"import generate_report; generate_report.generate_quality_report('{split_dir.replace(chr(92), '/')}', '{report_out.replace(chr(92), '/')}')\"", cwd=dataset_code_dir)
        
    else:
        print("\n" + "="*60)
        print(" 1. USING EXISTING SYNTHETIC DATASET")
        print("="*60)

    print("\n" + "="*60)
    print(" 2. MODEL TRAINING (TRANSFER LEARNING -> FINE TUNING)")
    print("="*60)
    run_cmd("python train.py", cwd=training_code_dir)

    print("\n" + "="*60)
    print(" 3. EVALUATION & BENCHMARKING")
    print("="*60)
    run_cmd("python evaluate.py --split test", cwd=training_code_dir)
    
    if os.path.exists(benchmark_dir):
        run_cmd(f"python domain_shift_benchmark.py --lab_dir ./benchmark_data/lab --field_dir ./benchmark_data/field", cwd=training_code_dir)
    else:
        print(f"\n⚠️ Skipping Domain Shift Benchmark: Benchmark directory not found at {benchmark_dir}")

    print("\n" + "="*60)
    print(" 4. QUANTIZATION SANITY CHECK (Keras float32 vs INT8 TFLite)")
    print("="*60)
    run_cmd("python check_quantization.py", cwd=training_code_dir)

    print("\n" + "="*60)
    print(" [SUCCESS] ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f" -> TFLite Model Exported: {os.path.join(training_code_dir, 'exported_model')}")
    print(f" -> Reports & Matrices Output: {reports_dir}")

if __name__ == "__main__":
    main()
