import os
import time
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

app = Flask(__name__)

@app.route('/')
def health_check():
    return {"status": "online", "bot": "BTC-USD Scalper", "timestamp": time.time()}

# Credentials read from environment variables on Render
API_KEY = os.environ.get("APACA_API_KEY", "PKHT7JZKIQYQARUAQC3VHEK3VW")
SECRET_KEY = os.environ.get("APACA_SECRET_KEY", "EHbhfoetyHhjPdv5jb5tdJ1WjgaUsVSQmjuS9VJ9cdKn")

SYMBOL_yf = "BTC-USD"
SYMBOL_alpaca = "BTC/USD"
QTY = 0.002
CHECK_INTERVAL_SECONDS = 300  # 5 minutes

STOP_LOSS_PCT = 0.015   # 1.5% Stop Loss
TAKE_PROFIT_PCT = 0.03  # 3.0% Take Profit

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_trading_bot():
    print("Background Trading Engine Initialized...")
    entry_price = None

    while True:
        try:
            print(f"\n--- Running Cycle at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            df = yf.download(SYMBOL_yf, period="5d", interval="5m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df["Return"] = df["Close"].pct_change()
            df["MA_Fast"] = df["Close"].rolling(window=5).mean()
            df["MA_Slow"] = df["Close"].rolling(window=20).mean()
            df["RSI"] = calculate_rsi(df["Close"], window=14)
            df["Vol_MA"] = df["Volume"].rolling(window=20).mean()
            df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
            df = df.dropna()

            feature_cols = ["Return", "MA_Fast", "MA_Slow", "RSI"]
            X = df[feature_cols]
            y = df["Target"]

            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_prec = precision_score(y_test, y_test_pred, zero_division=0) * 100
            print(f"[TEST EVAL] BUY Win Rate (Precision): {test_prec:.2f}%")

            live_pred = model.predict(X.iloc[[-1]])[0]
            current_price = float(df["Close"].iloc[-1])
            current_rsi = float(df["RSI"].iloc[-1])

            positions = trading_client.get_all_positions()
            has_position = any(p.symbol == SYMBOL_alpaca for p in positions)

            if has_position and entry_price is not None:
                pnl_pct = (current_price - entry_price) / entry_price
                print(f"[POSITION] Current PnL: {pnl_pct * 100:.2f}%")
                if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT:
                    print(f"Target reached ({pnl_pct*100:.2f}%). Exiting position...")
                    trading_client.close_position(SYMBOL_alpaca)
                    entry_price = None

            elif live_pred == 1 and not has_position and current_rsi < 70:
                print(f"Executing BUY order for {QTY} {SYMBOL_alpaca} at ${current_price:.2f}...")
                order = trading_client.submit_order(
                    order_data=MarketOrderRequest(
                        symbol=SYMBOL_alpaca,
                        qty=QTY,
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC
                    )
                )
                entry_price = current_price
                print(f"Order successfully sent! ID: {order.id}")
            else:
                print("No order condition met. Monitoring next interval...")

        except Exception as e:
            print(f"Trading loop error: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

# Launch background trading thread
t = threading.Thread(target=run_trading_bot, daemon=True)
t.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)