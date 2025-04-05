import pandas as pd
from get_tick_from_OHLC import generate_ticks_from_ohlcv
from ml_trend_predictor import train_trend_predictor
from grid_simulator_with_ml import SimpleGridSimulator
import pickle

# === Загрузка данных ===
df = pd.read_csv("solana-20250322143418221.csv")  # замените на свой путь к OHLCV-файлу

# === Генерация тиков ===
tick_df = generate_ticks_from_ohlcv(df)
prices = tick_df['price'].values

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

# === Обучение модели ===
df['Close'] = df['priceClose']  # временная адаптация под train_trend_predictor
model, scaler, df_ml, acc, conf, report = train_trend_predictor(df)

# === Подготовка признаков на тиках ===
# Здесь можно использовать признаки из df_ml по ближайшим временам или просто повторить последнюю строку как заглушку
features_df = pd.DataFrame([df_ml.drop(columns=['trend', 'Future']).iloc[-1].values] * len(prices),
                           columns=df_ml.drop(columns=['trend', 'Future']).columns)

# === Симуляция ===
sim = SimpleGridSimulator(
    prices=prices,
    ema_values=ema_values,
    features_df=features_df,
    model=model,
    scaler=scaler
)
sim.simulate()
sim.plot()

# === Сохранение модели и симуляции ===
with open("grid_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

with open("grid_simulation_with_ml.pkl", "wb") as f:
    pickle.dump(sim, f)
