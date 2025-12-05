"""
Main entry point for running a ProFiT optimization.
"""
import os
import sys

# Ensure the project root is in the python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import profit

def main():
    """
    Main execution function.
    """
    print("--- ProFiT Framework: Reproducing Research ---")
    
    # 1. Load configuration from file
    config = profit.load_config("config.yaml")
    print(f"Configuration loaded. LLM Client: {config.llm.client}")

    # 2. Create and run the optimizer
    # Using Fold 5 (Demo Fold) to match sample data
    print("Initializing optimizer (Fold: 5, Seed: 0)...")
    try:
        optimizer = profit.Optimizer(config, fold_idx=5, seed_idx=0)
        
        print("Running optimization loop...")
        results = optimizer.run()

        # 3. Generate the final report
        print("Optimization complete. Generating report...")
        report_dir = "runs/reproduction_run"
        results.generate_report(output_dir=report_dir, objectives=config.objectives)
        print(f"Report generated in '{report_dir}'.")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
