#!/usr/bin/env python
"""
Script to run hyperparameter tuning experiments in a sensible order:
1. Transformer experiments (run first):
   - One for top 10 (subset)
   - One for top 100 (subset)
2. XGBoost experiments:
   For each of three dataset configurations:
      (a) Subset top10, (b) Subset top100, and (c) Full with subsample fraction 0.05,
   run three experiments:
      i.   XGBoost with group "vol"
      ii.  XGBoost with group "ret"
      iii. XGBoost with group "vol" and asymmetric loss enabled.
      
Adjust the command‐line arguments (such as experiment names, mode, test_n, subsample fraction, etc.) as needed.
"""

import subprocess
import sys


def run_experiment(cmd_args):
    """Run a command (list of arguments) and print the command first."""
    cmd_str = " ".join(str(arg) for arg in cmd_args)
    print(f"\n=== Running: {cmd_str} ===\n")
    try:
        result = subprocess.run(cmd_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd_str}")
        sys.exit(e.returncode)


def run_transformer_experiments():
    # Transformer experiments (model_type "transformer") always run in sequence mode.
    # Run for top 10 (subset with test_n=10) and top 100.
    print("==== Running Transformer Experiments ====")
    
    # Transformer top10:
    cmd = [
        "python", "evaluation/hyperparameter_tuning.py",
        "-m", "transformer",
        "-d", "subset",
        "-t", "10",
        "--experiment-name", "transformer_top10"
    ]
    run_experiment(cmd)

    # Transformer top100:
    cmd = [
        "python", "evaluation/hyperparameter_tuning.py",
        "-m", "transformer",
        "-d", "subset",
        "-t", "100",
        "--experiment-name", "transformer_top100"
    ]
    run_experiment(cmd)


def run_xgboost_experiments():
    print("==== Running XGBoost Experiments ====")
    # Define dataset configurations for XGBoost experiments:
    # Each tuple is (config_name, config_params)
    dataset_configs = [
        ("top10", {"mode": "subset", "test_n": "10"}),
        ("top100", {"mode": "subset", "test_n": "100"}),
        ("whole", {"mode": "full", "subsample_fraction": "0.05"})
    ]

    # For each dataset configuration, run for group "vol" and "ret"
    for config_name, params in dataset_configs:
        for group in ["vol", "ret"]:
            exp_name = f"xgboost_{config_name}_{group}"
            cmd = [
                "python", "evaluation/hyperparameter_tuning.py",
                "-m", "xgboost",
                "-g", group,
                "--experiment-name", exp_name,
                "-d", params["mode"]
            ]
            # For subset mode, add test_n; for full, add subsample_fraction.
            if params["mode"] == "subset":
                cmd.extend(["-t", params["test_n"]])
            elif params["mode"] == "full":
                cmd.extend(["-f", params["subsample_fraction"]])
            run_experiment(cmd)

        # Now run the asymmetric loss version (only for group "vol")
        exp_name = f"xgboost_{config_name}_vol_asym"
        cmd = [
            "python", "evaluation/hyperparameter_tuning.py",
            "-m", "xgboost",
            "-g", "vol",
            "-a",  # asymmetric loss flag
            "--experiment-name", exp_name,
            "-d", params["mode"]
        ]
        if params["mode"] == "subset":
            cmd.extend(["-t", params["test_n"]])
        elif params["mode"] == "full":
            cmd.extend(["-f", params["subsample_fraction"]])
        run_experiment(cmd)


def main():
    print("Starting hyperparameter tuning experiments...\n")
    
    # Run transformer experiments first.
    run_transformer_experiments()
    
    # Then run XGBoost experiments.
    run_xgboost_experiments()
    
    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
