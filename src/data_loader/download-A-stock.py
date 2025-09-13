import tushare as ts
import dotenv
import os

dotenv.load_dotenv()
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")


def get_single_date_data(ts_code="600519.SH", trade_date="20240101", freq="5min"):
    pro = ts.pro_api(TUSHARE_TOKEN)
    # df = pro.stk_mins(ts_code=ts_code, trade_date=trade_date, freq=freq)
    df = pro.stk_mins(
        ts_code=ts_code, freq="5min", start_date="20250101", end_date="20250601"
    )
    df.to_csv(f"./data/min_{ts_code.split(".")[0]}_demo.csv")
    return df


if __name__ == "__main__":
    df = get_single_date_data()
    print(len(df))
