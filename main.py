"""
Main entry point for running a ProFiT optimization.
"""
import os
import profit

# The 'main.py' in the user's CWD is not in a package, so it can't find 'profit'
# without this path modification.
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    """
    Main execution function.
    """
    print("--- ProFiT Framework ---")
    
    # 1. Load configuration from file
    config = profit.load_config("config.yaml")
    print("Configuration loaded.")

    # 2. Create and run the optimizer
    print("Initializing optimizer...")
    optimizer = profit.Optimizer(config)
    
    print("Running optimization...")
    results = optimizer.run()

    # 3. Generate the final report
    print("Optimization complete. Generating report...")
    report_dir = "runs/my_first_run"
    results.generate_report(output_dir=report_dir, objectives=config.objectives)
    print(f"Report generated in '{report_dir}'.")


if __name__ == "__main__":
    main()