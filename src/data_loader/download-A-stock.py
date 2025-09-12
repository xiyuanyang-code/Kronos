import tushare as ts
import dotenv
import os

dotenv.load_dotenv()
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN")


def get_single_date_data(ts_code="600519.SH", trade_date="20240101", freq="5min"):
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.stk_mins(ts_code=ts_code, trade_date=trade_date, freq=freq)
    return df


if __name__ == "__main__":
    df = get_single_date_data()

    df.to_csv("./data/demo.csv")
