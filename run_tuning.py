#!/usr/bin/env python3
"""
Script to run automatic parameter tuning for Kronos model
"""

import os
import sys
import subprocess
import json
import time

# Add project root to path
sys.path.append(os.getcwd())


def run_tuning():
    """Run parameter tuning with different configurations"""

    # Define parameter sets to test
    param_sets = [
        {
            "name": "default",
            "params": {
                "lookback": 400,
                "pred_len": 120,
                "T": 1.0,
                "top_p": 0.9,
                "sample_count": 10,
            },
        },
        {
            "name": "conservative",
            "params": {
                "lookback": 600,
                "pred_len": 60,
                "T": 0.5,
                "top_p": 0.8,
                "sample_count": 15,
            },
        },
        {
            "name": "aggressive",
            "params": {
                "lookback": 200,
                "pred_len": 180,
                "T": 1.5,
                "top_p": 0.95,
                "sample_count": 5,
            },
        },
        {
            "name": "short_term",
            "params": {
                "lookback": 300,
                "pred_len": 30,
                "T": 0.8,
                "top_p": 0.85,
                "sample_count": 12,
            },
        },
        {
            "name": "long_term",
            "params": {
                "lookback": 500,
                "pred_len": 200,
                "T": 1.2,
                "top_p": 0.92,
                "sample_count": 8,
            },
        },
    ]

    results = []

    for param_set in param_sets:
        name = param_set["name"]
        params = param_set["params"]

        print(f"\nRunning {name} configuration...")
        print(f"Parameters: {params}")

        # Create a temporary script with these parameters
        script_content = f"""
import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from model import Kronos, KronosPredictor, KronosTokenizer
import time

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# Initialize the predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

df = pd.read_csv("./data/XSHG_5min_600977.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])

# Parameters from tuning set
lookback = {params["lookback"]}
pred_len = {params["pred_len"]}

# Prepare inputs for the predictor
x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.loc[: lookback - 1, "timestamps"]
y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

try:
    # Generate predictions
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T={params["T"]},
        top_p={params["top_p"]},
        sample_count={params["sample_count"]},
    )
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results/prediction_{{timestamp}}_{name}.csv"
    pred_df.to_csv(filename)
    print(f"Results saved to {{filename}}")
    
    # Calculate some basic metrics
    mean_close = pred_df["close"].mean()
    print(f"Mean close price: {{mean_close:.4f}}")
    
except Exception as e:
    print(f"Error running {{name}} configuration: {{str(e)}}")
"""

        # Write temporary script
        script_path = f"/tmp/tuning_{name}.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Run the script
        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,
            )  # 5 minute timeout
            end_time = time.time()

            if result.returncode == 0:
                print(
                    f"Successfully completed {name} configuration in {end_time - start_time:.2f} seconds"
                )
                results.append(
                    {
                        "name": name,
                        "params": params,
                        "status": "success",
                        "time": end_time - start_time,
                        "output": result.stdout,
                    }
                )
            else:
                print(f"Failed to run {name} configuration")
                print(f"Error: {result.stderr}")
                results.append(
                    {
                        "name": name,
                        "params": params,
                        "status": "failed",
                        "error": result.stderr,
                    }
                )
        except subprocess.TimeoutExpired:
            print(f"Timeout running {name} configuration (exceeded 5 minutes)")
            results.append({"name": name, "params": params, "status": "timeout"})
        except Exception as e:
            print(f"Error running {name} configuration: {str(e)}")
            results.append(
                {"name": name, "params": params, "status": "error", "error": str(e)}
            )

        # Clean up temporary script
        if os.path.exists(script_path):
            os.remove(script_path)

        # Small delay between runs
        time.sleep(5)

    # Save results summary
    with open("results/tuning_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTuning completed. Summary saved to results/tuning_summary.json")

    # Print summary
    print("\nSummary:")
    for result in results:
        if result["status"] == "success":
            print(f"  {result['name']}: SUCCESS ({result['time']:.2f}s)")
        else:
            print(f"  {result['name']}: {result['status'].upper()}")


if __name__ == "__main__":
    run_tuning()
