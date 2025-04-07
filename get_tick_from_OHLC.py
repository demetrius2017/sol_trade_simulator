import pandas as pd

def generate_ticks_from_ohlcv(df):
    """
    Преобразует OHLCV-данные в тиковые события: Open -> High/Low -> Low/High -> Close
    Восстанавливает вероятную последовательность событий на основе временных меток.
    Возвращает DataFrame с колонками: timestamp, price, source (Open/High/Low/Close)
    """
    ticks = []

    for _, row in df.iterrows():
        # Добавляем Open
        ticks.append({
            'timestamp': row['Open time'],
            'price': row['Open'],
            'source': 'open'
        })

        # Определяем последовательность high/low
        if row['High'] > row['Low']:
            ticks.append({'timestamp': row['Open time'], 'price': row['High'], 'source': 'high'})
            ticks.append({'timestamp': row['Open time'], 'price': row['Low'],  'source': 'low'})
        else:
            ticks.append({'timestamp': row['Open time'], 'price': row['Low'],  'source': 'low'})
            ticks.append({'timestamp': row['Open time'], 'price': row['High'], 'source': 'high'})

        # Добавляем Close
        ticks.append({
            'timestamp': row['Open time'],
            'price': row['Close'],
            'source': 'close'
        })

    tick_df = pd.DataFrame(ticks)
    tick_df.sort_values('timestamp', inplace=True)
    tick_df.reset_index(drop=True, inplace=True)

    return tick_df

def load_tick_data(filepath):
    """
    Загружает тиковые данные из файла и преобразует timestamp из наносекунд в миллисекунды.
    """
    columns = ["trade_id", "price", "qty", "quote_qty", "timestamp", "is_buyer_maker", "is_best_match"]
    data = pd.read_csv(filepath, names=columns)
    data['timestamp'] = data['timestamp'] // 1_000_000  # Преобразование в миллисекунды
    return data
