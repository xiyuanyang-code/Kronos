import os
import sys

sys.path.append(os.getcwd())

import pandas as pd
import json
import time
from model import Kronos, KronosPredictor, KronosTokenizer

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# Initialize the predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

df = pd.read_csv("./data/XSHG_5min_600977.csv")
df["timestamps"] = pd.to_datetime(df["timestamps"])

# Define context window and prediction length
# These are the parameters you can adjust for tuning
lookback = 400      # Number of historical data points to consider
pred_len = 120      # Number of future time points to predict

# Prepare inputs for the predictor
x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.loc[: lookback - 1, "timestamps"]
y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

# Generate predictions
# These are the key parameters for tuning model behavior:
# T: Temperature for sampling (higher = more random, lower = more deterministic)
# top_p: Nucleus sampling probability (higher = more diverse, lower = more focused)
# sample_count: Number of forecast paths to generate and average (higher = more stable but slower)
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature for sampling
    top_p=0.9,      # Nucleus sampling probability
    sample_count=10, # Number of forecast paths to generate and average
)

print("Forecasted Data Head:")
print(pred_df.head())

timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"results/prediction_{timestamp}.csv"
pred_df.to_csv(filename)
print(f"Results saved to {filename}")


# Example of how to use the function for parameter tuning:
# Load parameters from config
# params = load_params("conservative")  # or "aggressive", "balanced", "default"
# pred_df = run_with_params(
#     lookback_val=params["lookback"], 
#     pred_len_val=params["pred_len"], 
#     T_val=params["T"], 
#     top_p_val=params["top_p"], 
#     sample_count_val=params["sample_count"]
# )
# if pred_df is not None:
#     print("Custom parameter run results:")
#     print(pred_df.head())
