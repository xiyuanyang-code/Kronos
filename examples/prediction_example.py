import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("../")
sys.path.append(os.getcwd())
from model import Kronos, KronosTokenizer, KronosPredictor
from datetime import datetime

time_stamp = datetime.now().strftime("%m%d%H%M%S")


def plot_prediction(kline_df, pred_df, model_name):
    pred_df.index = kline_df.index[-pred_df.shape[0] :]
    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "Ground Truth"
    sr_pred_volume.name = "Prediction"

    sr_open = kline_df["open"]
    sr_pred_open = pred_df["open"]
    sr_open.name = "Ground Truth"
    sr_pred_open.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)
    open_df = pd.concat([sr_open, sr_pred_open], axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax1.plot(
        close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(
        volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    ax3.plot(
        open_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax3.plot(open_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax3.set_ylabel("Open", fontsize=14)
    ax3.legend(loc="upper left", fontsize=12)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"./image/{model_name}_{time_stamp}.png")


def experiment(model, model_name):
    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

    # 3. Prepare Data
    df = pd.read_csv("./data/demo_600519.csv")
    df["timestamps"] = pd.to_datetime(df["trade_date"])

    lookback = 400
    pred_len = 120

    x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: lookback - 1, "timestamps"]
    y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

    # 4. Make Prediction
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    # Combine historical and forecasted data for plotting
    kline_df = df.loc[: lookback + pred_len - 1]

    # visualize
    plot_prediction(kline_df, pred_df, model_name)


if __name__ == "__main__":
    # setting environment for cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # 1. Load Model and Tokenizer
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    # model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

    models = [
        ("kronos-small", Kronos.from_pretrained("NeoQuasar/Kronos-small")),
        ("kronos-base", Kronos.from_pretrained("NeoQuasar/Kronos-base")),
    ]

    for model_single in models:
        model = model_single[1]
        model_name = model_single[0]
        print(f"Using {model_name} for testing")
        experiment(model=model, model_name=model_name)
