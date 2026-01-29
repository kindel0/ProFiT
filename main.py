"""
Main entry point for running a ProFiT optimization.
"""
import argparse
import os
import sys

# Ensure the project root is in the python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import profit
from profit.batch import BatchRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ProFiT: Program Search for Financial Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run (demo fold, seed 0)
  python main.py --fold 5 --seed 0

  # Run all 5 folds with seed 0
  python main.py --folds 0,1,2,3,4 --seed 0

  # Run fold 0 with all 5 seeds
  python main.py --fold 0 --seeds 0,1,2,3,4

  # Full batch: 5 folds x 5 seeds = 25 runs
  python main.py --folds 0,1,2,3,4 --seeds 0,1,2,3,4

  # Parallel execution
  python main.py --folds 0,1,2,3,4 --seeds 0,1,2,3,4 --parallel
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Single fold index to run (0-4, or 5 for demo fold)"
    )

    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated list of fold indices (e.g., '0,1,2,3,4')"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed strategy index to run (0-4)"
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seed indices (e.g., '0,1,2,3,4')"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run fold/seed combinations in parallel using Ray"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="runs",
        help="Output directory for results (default: runs)"
    )

    return parser.parse_args()


def parse_list(value: str) -> list:
    """Parse comma-separated list of integers."""
    return [int(x.strip()) for x in value.split(",")]


def main():
    """Main execution function."""
    args = parse_args()

    print("--- ProFiT Framework: Reproducing Research ---")

    # 1. Load configuration from file
    config = profit.load_config(args.config)
    print(f"Configuration loaded. LLM Client: {config.llm.client}")

    # 2. Determine folds and seeds to run
    if args.folds:
        folds = parse_list(args.folds)
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = [5]  # Default: demo fold

    if args.seeds:
        seeds = parse_list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [0]  # Default: seed 0

    # 3. Run optimization
    try:
        is_batch = len(folds) > 1 or len(seeds) > 1

        if is_batch:
            # Batch mode: multiple folds and/or seeds
            runner = BatchRunner(
                config=config,
                folds=folds,
                seeds=seeds,
                parallel=args.parallel
            )
            batch_results = runner.run_all()

            # Generate batch report
            print("\nOptimization complete. Generating batch report...")
            report_dir = os.path.join(args.output, "batch_run")
            batch_results.generate_batch_report(report_dir)

            # Print summary
            agg = batch_results.aggregate_metrics()
            print("\n" + "="*60)
            print("BATCH SUMMARY")
            print("="*60)
            print(f"Runs: {agg.get('successful_runs', 0)}/{agg.get('num_runs', 0)} successful")
            if agg.get('test_fitness_mean') is not None:
                print(f"Test Fitness: {agg['test_fitness_mean']:.2f}% +/- {agg['test_fitness_std']:.2f}%")
            if agg.get('test_sharpe_mean') is not None:
                print(f"Test Sharpe:  {agg['test_sharpe_mean']:.2f} +/- {agg['test_sharpe_std']:.2f}")

        else:
            # Single run mode
            fold_idx = folds[0]
            seed_idx = seeds[0]

            print(f"Initializing optimizer (Fold: {fold_idx}, Seed: {seed_idx})...")
            optimizer = profit.Optimizer(config, fold_idx=fold_idx, seed_idx=seed_idx)

            print("Running optimization loop...")
            results = optimizer.run()

            # Generate report
            print("Optimization complete. Generating report...")
            report_dir = os.path.join(args.output, f"fold{fold_idx}_seed{seed_idx}")
            results.generate_report(output_dir=report_dir, objectives=config.objectives)
            print(f"Report generated in '{report_dir}'.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
