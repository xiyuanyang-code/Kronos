import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from typing import Optional, Dict

sys.path.append("../")
sys.path.append(os.getcwd())
from model import Kronos, KronosTokenizer, KronosPredictor


class KronosStockPredictor:
    """
    A class to handle stock data prediction using Kronos models.
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cuda:0",
        max_context: int = 512,
        data_path: str = "./data/demo_600519.csv",
        stock_code: str = None,
    ):
        """
        Initializes the predictor with specified model and tokenizer.

        Args:
            model_name (str): Name or path of the pre-trained Kronos model.
            tokenizer_name (str): Name or path of the pre-trained Kronos tokenizer.
            device (str): The device to run the model on (e.g., 'cuda:0', 'cpu').
            max_context (int): The maximum context length for the predictor.
            data_path (str): Path to the stock data CSV file.
        """
        if stock_code is None:
            print("Please pass stock code!")
            stock_code = "DEFAULT"
        stock_code = stock_code.replace(".", "_")

        self.device = device
        self.max_context = max_context
        self.data_path = data_path
        self.stock_code = stock_code
        self.model_name = model_name

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        print(f"Loading model: {model_name}")
        self.model = Kronos.from_pretrained(model_name)

        self.predictor = KronosPredictor(
            self.model, self.tokenizer, device=self.device, max_context=self.max_context
        )

    def load_data(self) -> Optional[pd.DataFrame]:
        """Loads and preprocesses stock data from the specified CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            df["timestamps"] = pd.to_datetime(df["trade_date"])
            return df
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return None

    def plot_prediction(
        self,
        kline_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        output_path: str = "./image",
    ):
        """
        Plots the ground truth and predicted stock data.

        Args:
            kline_df (pd.DataFrame): The combined historical and ground truth data.
            pred_df (pd.DataFrame): The predicted data.
            output_path (str): Directory to save the plot.
        """
        # Set index for pred_df to align with kline_df for plotting
        pred_df.index = kline_df.index[-pred_df.shape[0] :]

        # Prepare data for plotting
        close_df = pd.concat([kline_df["close"], pred_df["close"]], axis=1)
        volume_df = pd.concat([kline_df["volume"], pred_df["volume"]], axis=1)
        open_df = pd.concat([kline_df["open"], pred_df["open"]], axis=1)

        close_df.columns = ["Ground Truth", "Prediction"]
        volume_df.columns = ["Ground Truth", "Prediction"]
        open_df.columns = ["Ground Truth", "Prediction"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        ax1.plot(
            close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
        )
        ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
        ax1.set_ylabel("Close Price", fontsize=14)
        ax1.legend(loc="lower left", fontsize=12)
        ax1.grid(True)
        ax1.set_title(f"Prediction for {self.model_name}", fontsize=16)

        ax2.plot(
            volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
        )
        ax2.plot(
            volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5
        )
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

        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        plt.savefig(
            f"{output_path}/{(self.model_name).split("/")[-1]}_{time_stamp}_{self.stock_code}.png"
        )
        print(
            f"Plot saved to {output_path}/{(self.model_name).split("/")[-1]}_{time_stamp}.png"
        )
        plt.close(fig)

    def predict(
        self,
        lookback: int = 400,
        pred_len: int = 120,
        pred_params: Optional[Dict] = None,
    ):
        """
        Main function to prepare data, make prediction, and plot results.

        Args:
            lookback (int): The number of historical data points to use for context.
            pred_len (int): The number of future data points to predict.
            pred_params (Optional[Dict]): A dictionary of parameters for the predictor.
        """

        # 3. Prepare Data
        df = self.load_data()
        if df is None:
            return

        if len(df) < lookback + pred_len:
            print(
                "Error: Not enough data for the specified lookback and prediction length."
            )
            return

        x_df = df.loc[
            : lookback - 1, ["open", "high", "low", "close", "volume", "amount"]
        ]
        x_timestamp = df.loc[: lookback - 1, "timestamps"]
        y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

        # 4. Make Prediction with customizable parameters
        default_pred_params = {
            "pred_len": pred_len,
            "T": 1.0,
            "top_p": 0.9,
            "sample_count": 1,
            "verbose": True,
        }
        # Merge default and user-provided parameters
        if pred_params:
            default_pred_params.update(pred_params)

        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            **default_pred_params,
        )

        # Combine historical and forecasted data for plotting
        kline_df = df.loc[: lookback + pred_len - 1]

        self.plot_prediction(kline_df, pred_df)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Define a list of models and their names to iterate through
    models_to_test = [
        "NeoQuasar/Kronos-small",
        "NeoQuasar/Kronos-base",
    ]

    # Test each model
    for model_path in models_to_test:
        print(f"\n--- Testing with model: {model_path} ---")
        try:
            # Instantiate the class with a specific model
            predictor_instance = KronosStockPredictor(
                model_name=model_path, stock_code="600519.SH"
            )

            # Use the predict method to run the full experiment
            # You can pass different parameters here for each experiment
            predictor_instance.predict(
                lookback=400,
                pred_len=120,
                pred_params={"T": 1.0, "top_p": 0.9, "sample_count": 1},
            )
        except Exception as e:
            print(f"An error occurred with model {model_path}: {e}")
