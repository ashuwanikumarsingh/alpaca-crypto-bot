import os
import time
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
import gradio as gr

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

app = Flask(__name__)

API_KEY = os.environ.get("APACA_API_KEY", "PKHT7JZKIQYQARUAQC3VHEK3VW")
SECRET_KEY = os.environ.get("APACA_SECRET_KEY", "EHbhfoetyHhjPdv5jb5tdJ1WjgaUsVSQmjuS9VJ9cdKn")

SYMBOL_yf = "BTC-USD"
SYMBOL_alpaca = "BTC/USD"
QTY = 0.002
CHECK_INTERVAL_SECONDS = 300

STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.03

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Shared state dictionary for live dashboard updates
bot_state = {
    "last_cycle": "Initializing...",
    "price": "Loading...",
    "rsi": "Loading...",
    "signal": "Loading...",
    "win_rate": "Calculating...",
    "entry_price": None
}

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_trading_bot():
    while True:
        try:
            bot_state["last_cycle"] = time.strftime('%Y-%m-%d %H:%M:%S')
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
            bot_state["win_rate"] = f"{test_prec:.2f}% (over {int(np.sum(y_test_pred == 1))} signals)"

            live_pred = model.predict(X.iloc[[-1]])[0]
            current_price = float(df["Close"].iloc[-1])
            current_rsi = float(df["RSI"].iloc[-1])

            bot_state["price"] = f"${current_price:,.2f}"
            bot_state["rsi"] = f"{current_rsi:.2f}"
            bot_state["signal"] = "BUY (1)" if live_pred == 1 else "HOLD / WAIT (0)"

            positions = trading_client.get_all_positions()
            has_position = any(p.symbol == SYMBOL_alpaca for p in positions)

            if has_position and bot_state["entry_price"] is not None:
                pnl_pct = (current_price - bot_state["entry_price"]) / bot_state["entry_price"]
                if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT:
                    trading_client.close_position(SYMBOL_alpaca)
                    bot_state["entry_price"] = None

            elif live_pred == 1 and not has_position and current_rsi < 70:
                trading_client.submit_order(
                    order_data=MarketOrderRequest(
                        symbol=SYMBOL_alpaca,
                        qty=QTY,
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC
                    )
                )
                bot_state["entry_price"] = current_price

        except Exception as e:
            bot_state["last_cycle"] = f"Error: {e}"

        time.sleep(CHECK_INTERVAL_SECONDS)

# Launch background bot thread
t = threading.Thread(target=run_trading_bot, daemon=True)
t.start()

# --- GRADIO DASHBOARD INTERFACE ---
def get_live_metrics():
    try:
        account = trading_client.get_account()
        cash = f"${float(account.cash):,.2f}"
        equity = f"${float(account.portfolio_value):,.2f}"

        positions = trading_client.get_all_positions()
        pos_match = [p for p in positions if p.symbol == SYMBOL_alpaca]
        if pos_match:
            pos = pos_match[0]
            pos_info = f"{pos.qty} BTC @ ${float(pos.avg_entry_price):,.2f} | PnL: {float(pos.unrealized_plpc)*100:.2f}%"
        else:
            pos_info = "Flat (No active trade)"

        return (
            cash,
            equity,
            pos_info,
            bot_state["price"],
            bot_state["rsi"],
            bot_state["signal"],
            bot_state["win_rate"],
            bot_state["last_cycle"]
        )
    except Exception as e:
        return f"Error: {e}", "-", "-", "-", "-", "-", "-", "-"

def manual_buy():
    try:
        trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=SYMBOL_alpaca,
                qty=QTY,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC
            )
        )
        return "Manual BUY executed!"
    except Exception as e:
        return f"Buy failed: {e}"

def manual_close():
    try:
        trading_client.close_position(SYMBOL_alpaca)
        bot_state["entry_price"] = None
        return "Position closed."
    except Exception as e:
        return f"Close failed: {e}"

with gr.Blocks(title="Alpaca Crypto Terminal") as demo:
    gr.Markdown("# 🚀 Live Alpaca Algo-Trading Terminal")
    
    with gr.Row():
        cash_box = gr.Textbox(label="Available Cash", interactive=False)
        equity_box = gr.Textbox(label="Portfolio Value", interactive=False)
        pos_box = gr.Textbox(label="Open Position", interactive=False)

    with gr.Row():
        price_box = gr.Textbox(label="BTC Price", interactive=False)
        rsi_box = gr.Textbox(label="RSI (14)", interactive=False)
        signal_box = gr.Textbox(label="Bot Signal", interactive=False)

    with gr.Row():
        winrate_box = gr.Textbox(label="Test Win Rate (Out-of-Sample)", interactive=False)
        heartbeat_box = gr.Textbox(label="Last Strategy Cycle", interactive=False)

    with gr.Row():
        buy_btn = gr.Button("🟢 Instant BUY", variant="primary")
        close_btn = gr.Button("🔴 Flatten Position", variant="stop")
    
    action_feedback = gr.Textbox(label="Execution Status", interactive=False)

    buy_btn.click(fn=manual_buy, outputs=[action_feedback])
    close_btn.click(fn=manual_close, outputs=[action_feedback])

    # Poll live metrics every 5 seconds
    demo.load(
        fn=get_live_metrics,
        outputs=[cash_box, equity_box, pos_box, price_box, rsi_box, signal_box, winrate_box, heartbeat_box],
        every=5
    )

# Mount Gradio UI directly onto the root path of the Flask server
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)