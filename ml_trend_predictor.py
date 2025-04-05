
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def train_trend_predictor(df):
    X, y, df_ml = prepare_ml_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    conf = confusion_matrix(y_test, predictions, labels=[1, 0, -1])
    report = classification_report(y_test, predictions)

    return model, scaler, df_ml, acc, conf, report
