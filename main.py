
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
