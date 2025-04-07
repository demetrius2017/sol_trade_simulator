import requests
import pandas as pd
import time

def get_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Преобразование даты в миллисекунды
def date_to_milliseconds(date_str):
    return int(time.mktime(time.strptime(date_str, '%Y-%m-%d %H:%M:%S')) * 1000)

# Укажите начальную и конечную даты
start_date = '2024-01-01 00:00:00'
end_date = '2025-04-05 23:59:59'

start_time = date_to_milliseconds(start_date)
end_time = date_to_milliseconds(end_date)

symbol = 'SOLUSDT'
interval = '1m'

data = get_binance_klines(symbol, interval, start_time, end_time)

# Преобразование данных в DataFrame
df = pd.DataFrame(data, columns=[
    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close time', 'Quote asset volume', 'Number of trades',
    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
])

# Преобразование временных меток в читаемый формат
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

# Сохранение данных в CSV
df.to_csv('solana_minute_data.csv', index=False)
