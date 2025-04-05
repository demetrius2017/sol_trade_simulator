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
            'timestamp': row['timeOpen'],
            'price': row['priceOpen'],
            'source': 'open'
        })

        # Определяем последовательность high/low
        if row['timeHigh'] < row['timeLow']:
            ticks.append({'timestamp': row['timeHigh'], 'price': row['priceHigh'], 'source': 'high'})
            ticks.append({'timestamp': row['timeLow'],  'price': row['priceLow'],  'source': 'low'})
        else:
            ticks.append({'timestamp': row['timeLow'],  'price': row['priceLow'],  'source': 'low'})
            ticks.append({'timestamp': row['timeHigh'], 'price': row['priceHigh'], 'source': 'high'})

        # Добавляем Close
        ticks.append({
            'timestamp': row['timeClose'],
            'price': row['priceClose'],
            'source': 'close'
        })

    tick_df = pd.DataFrame(ticks)
    tick_df.sort_values('timestamp', inplace=True)
    tick_df.reset_index(drop=True, inplace=True)
    return tick_df
