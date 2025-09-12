import os
import sys

sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from model import Kronos, KronosPredictor, KronosTokenizer
import json
import time

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

df = pd.read_csv("./data/XSHG_5min_600977.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])

# Define parameter ranges for tuning
param_grid = {
    "lookback": [200, 400, 600],
    "pred_len": [60, 120],
    "T": [0.5, 1.0, 1.5],
    "top_p": [0.8, 0.9, 0.95],
    "sample_count": [5, 10, 15]
}

def run_prediction_with_params(lookback, pred_len, T, top_p, sample_count, device="cuda:0"):
    """Run prediction with specified parameters and return results"""
    try:
        # Initialize the predictor for each run to avoid memory issues
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        
        # Prepare inputs for the predictor
        x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = df.loc[: lookback - 1, "timestamps"]
        y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]
        
        # Check if we have enough data
        if len(x_df) < lookback or len(y_timestamp) < pred_len:
            return None, "Insufficient data"
        
        # Generate predictions
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
        )
        
        # Calculate some metrics (example: mean of close prices)
        mean_close = pred_df["close"].mean() if "close" in pred_df.columns else 0
        
        return pred_df, {
            "mean_close": mean_close,
            "lookback": lookback,
            "pred_len": pred_len,
            "T": T,
            "top_p": top_p,
            "sample_count": sample_count,
            "timestamp": time.strftime("%Y%m%d_%H%M%S")
        }
    except Exception as e:
        return None, f"Error: {str(e)}"

def grid_search():
    """Perform grid search over parameter combinations"""
    results = []
    
    total_combinations = (
        len(param_grid["lookback"]) *
        len(param_grid["pred_len"]) *
        len(param_grid["T"]) *
        len(param_grid["top_p"]) *
        len(param_grid["sample_count"])
    )
    
    print(f"Starting grid search with {total_combinations} combinations...")
    
    count = 0
    for lookback in param_grid["lookback"]:
        for pred_len in param_grid["pred_len"]:
            for T in param_grid["T"]:
                for top_p in param_grid["top_p"]:
                    for sample_count in param_grid["sample_count"]:
                        count += 1
                        print(f"Running combination {count}/{total_combinations}")
                        print(f"  lookback={lookback}, pred_len={pred_len}, T={T}, top_p={top_p}, sample_count={sample_count}")
                        
                        pred_df, metrics = run_prediction_with_params(
                            lookback, pred_len, T, top_p, sample_count
                        )
                        
                        if pred_df is not None and isinstance(metrics, dict):
                            # Save prediction results
                            filename = f"results/prediction_{metrics['timestamp']}_lb{lookback}_pl{pred_len}_T{T}_tp{top_p}_sc{sample_count}.csv"
                            pred_df.to_csv(filename, index=False)
                            
                            # Add to results
                            results.append(metrics)
                            print(f"  Success! Mean close price: {metrics['mean_close']:.4f}")
                        else:
                            print(f"  Failed: {metrics}")
    
    # Save results summary
    with open("results/tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Grid search completed. Results saved to results/tuning_results.json")
    return results

if __name__ == "__main__":
    grid_search()