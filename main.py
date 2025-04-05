import pandas as pd
import os
import numpy as np
from get_tick_from_OHLC import generate_ticks_from_ohlcv
from grid_simulator_with_ml_hedged import SimpleGridSimulator
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def prepare_ml_data(df, horizon=1, flat_range=0.01):
    df = df.copy()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['RSI'] = compute_rsi(df['Close'])
    df['Volatility'] = df['Close'].rolling(12).std()
    df['Return'] = df['Close'].pct_change()
    df['Future'] = df['Close'].shift(-horizon)
    df['trend'] = np.where(df['Future'] > df['Close'] * (1 + flat_range), 1,
                   np.where(df['Future'] < df['Close'] * (1 - flat_range), -1, 0))
    df.dropna(inplace=True)
    features = ['EMA12', 'EMA26', 'MACD', 'RSI', 'Volatility', 'Return']
    X = df[features]
    y = df['trend']
    return X, y, df

df = pd.read_csv("solana-20250322143418221.csv")
tick_file = "solana_tick_data.csv"

if os.path.exists(tick_file):
    tick_df = pd.read_csv(tick_file)
    tick_df['timestamp'] = pd.to_datetime(tick_df['timestamp'])
else:
    tick_df = generate_ticks_from_ohlcv(df)
    tick_df.to_csv(tick_file, index=False)

prices = tick_df['price'].values
timestamps = pd.date_range(start=tick_df['timestamp'].iloc[0], periods=len(prices), freq='s')

def compute_ema(prices, period):
    ema = []
    k = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i < period:
            ema.append(sum(prices[:i+1]) / (i+1))
        else:
            ema.append(price * k + ema[-1] * (1 - k))
    return ema

ema_values = compute_ema(prices, 20)

df['Close'] = df['priceClose']
X, y, df_ml = prepare_ml_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

features_df = X.reset_index(drop=True).iloc[:len(prices)].copy()
if len(features_df) < len(prices):
    last_row = features_df.iloc[-1]
    while len(features_df) < len(prices):
        features_df.loc[len(features_df)] = last_row

sim = SimpleGridSimulator(
    prices=prices,
    ema_values=ema_values,
    features_df=features_df,
    model=model,
    scaler=scaler,
    timestamps=timestamps
)
sim.simulate()
sim.plot()

with open("grid_simulation_with_ml.pkl", "wb") as f:
    pickle.dump(sim, f)

def extract_hedge_table(sim):
    data = []
    for h in sim.hedge_history:
        data.append({
            "Type": h.order_type,
            "Entry Price": h.entry_price,
            "Exit Price": h.exit_price,
            "Volume": h.volume,
            "Profit": h.floating_profit(h.exit_price) if h.exit_price else 0
        })
    df = pd.DataFrame(data)
    print("\n=== Hedge Events Table ===")
    print(df.to_string(index=False))
    print("==========================\n")
    return df

extract_hedge_table(sim)

# Вывод таблицы с логами ордеров
if hasattr(sim, 'order_log'):
    order_log_df = pd.DataFrame(sim.order_log)
    print("\n=== Order Log ===")
    print(order_log_df.to_string(index=False))
    print("=================\n")

# Вывод таблицы с логами хеджей
if hasattr(sim, 'hedge_log'):
    hedge_log_df = pd.DataFrame(sim.hedge_log)
    print("\n=== Hedge Log ===")
    print(hedge_log_df.to_string(index=False))
    print("==================\n")

    # Сохранение таблицы хеджей в файл
    hedge_log_file = "hedge_log.csv"
    hedge_log_df.to_csv(hedge_log_file, index=False)
    print(f"Hedge log saved to {hedge_log_file}")

# Добавление вывода итогового капитала, максимальной просадки и уплаченной комиссии
final_equity = sim.equity_history[-1]
peak = pd.Series(sim.equity_history).cummax()
max_drawdown = ((peak - sim.equity_history) / peak).max()

# Обновление расчёта общей комиссии с использованием нового метода calculate_commission
total_commission = sim.calculate_commission()

print("\n=== Simulation Summary ===")
print(f"Final Equity: {final_equity:.2f} USD")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Total Commission Paid: {total_commission:.2f} USD")
print("==========================\n")

# Сохранение деталей о текущем эквити, балансе и просадке в файл
summary_data = {
    "Final Equity": final_equity,
    "Final Balance": sim.balance,
    "Max Drawdown": max_drawdown,
    "Total Commission Paid": total_commission
}
summary_file = "simulation_summary.csv"
with open(summary_file, "w") as f:
    for key, value in summary_data.items():
        f.write(f"{key}: {value}\n")

print(f"Simulation summary saved to {summary_file}")

# Сохранение истории ордеров с текущим балансом, эквити и просадкой в файл
if hasattr(sim, 'order_log'):
    order_log_df = pd.DataFrame(sim.order_log)

    # Убедимся, что длины совпадают, добавляя только те записи, которые соответствуют длине order_log
    min_length = min(len(order_log_df), len(sim.balance_history), len(sim.equity_history))
    order_log_df = order_log_df.iloc[:min_length]
    order_log_df['Balance'] = sim.balance_history[:min_length]
    order_log_df['Equity'] = sim.equity_history[:min_length]
    order_log_df['Drawdown'] = [sim.get_drawdown(e) for e in sim.equity_history[:min_length]]

    detailed_summary_file = "detailed_simulation_summary.csv"
    order_log_df.to_csv(detailed_summary_file, index=False)

    print(f"Detailed simulation summary saved to {detailed_summary_file}")
