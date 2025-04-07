import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class Position:
    def __init__(self, order_type, entry_price, volume):
        self.order_type = order_type
        self.entry_price = entry_price
        self.volume = volume
        self.closed = False
        self.exit_price = None

    def floating_profit(self, current_price):
        if self.order_type == "buy":
            return (current_price - self.entry_price) * self.volume
        else:
            return (self.entry_price - current_price) * self.volume

    def close(self, exit_price, taker_fee=0.0004, maker_fee=0.0002):
        self.exit_price = exit_price
        self.closed = True

        open_fee = self.entry_price * self.volume * maker_fee
        close_fee = exit_price * self.volume * taker_fee
        gross_profit = self.floating_profit(exit_price)
        net_profit = gross_profit - open_fee - close_fee

        # Учет комиссии в расчете прибыли
        self.profit = net_profit
        return net_profit


class SimpleGridSimulator:
    def __init__(self, prices, ema_values, features_df, model, scaler,
                 initial_balance=100000, grid_step=0.001, grid_size=70,
                 direction_change_threshold=0.1, timestamps=None):
        self.prices = prices
        self.ema_values = ema_values
        self.features_df = features_df
        self.model = model
        self.scaler = scaler
        self.timestamps = timestamps if timestamps is not None else list(range(len(prices)))

        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.grid_step = grid_step
        self.grid_size = grid_size
        self.direction_change_threshold = direction_change_threshold
        self.positions = []
        self.grid_center = None
        self.grid = []
        self.direction = "neutral"
        self.max_equity = initial_balance

        self.equity_history = []
        self.balance_history = []
        self.direction_history = []
        self.hedge_position = None
        self.hedge_history = []

    def generate_grid(self, center_price):
        grid = []
        for i in range(1, self.grid_size + 1):
            delta = self.grid_step * i
            grid.append(("buy", center_price * (1 - delta)))
            grid.append(("sell", center_price * (1 + delta)))
        return grid

    def volume_by_direction(self, order_type):
        if self.direction == "neutral":
            return 1
        if self.direction == "long":
            return 2 if order_type == "buy" else 1
        if self.direction == "short":
            return 1 if order_type == "buy" else 2

    def log_order(self, order_type, entry_price, volume, exit_price=None, profit=0):
        if not hasattr(self, 'order_log'):
            self.order_log = []
        self.order_log.append({
            "Type": order_type,
            "Entry Price": entry_price,
            "Volume": volume,
            "Exit Price": exit_price,
            "Profit": profit
        })

    def log_hedge(self, order_type, entry_price, volume, exit_price=None, profit=0):
        if not hasattr(self, 'hedge_log'):
            self.hedge_log = []
        self.hedge_log.append({
            "Type": order_type,
            "Entry Price": entry_price,
            "Volume": volume,
            "Exit Price": exit_price,
            "Profit": profit
        })

    def find_target_exit_price(self, position, current_price):
        """
        Определяет целевую цену закрытия для позиции на основе сетки и текущей цены.
        """
        if position.order_type == "buy":
            # Найти ближайший sell-уровень выше entry_price, но не выше текущей цены
            target_prices = [price for order_type, price in self.grid if order_type == "sell" and position.entry_price < price <= current_price]
        else:
            # Найти ближайший buy-уровень ниже entry_price, но не ниже текущей цены
            target_prices = [price for order_type, price in self.grid if order_type == "buy" and position.entry_price > price >= current_price]

        return min(target_prices, default=None) if position.order_type == "buy" else max(target_prices, default=None)

    def simulate(self):
        print(f"Начало симуляции: {len(self.prices)} ценовых точек, {len(self.ema_values)} значений EMA")
        
        # Преобразуем prices и ema_values в обычные списки, если они являются массивами NumPy
        if isinstance(self.prices, np.ndarray):
            self.prices = self.prices.tolist()
        if isinstance(self.ema_values, np.ndarray):
            self.ema_values = self.ema_values.tolist()
        
        # Проверка: если ema_values пусто, используем цены как центр сетки
        if not self.ema_values and self.prices:
            print("EMA значения не предоставлены, используем цены как центр сетки")
            self.ema_values = self.prices.copy()
        
        # Проверка длин массивов
        if len(self.ema_values) != len(self.prices):
            print(f"Предупреждение: длины массивов не совпадают. Цены: {len(self.prices)}, EMA: {len(self.ema_values)}")
            # Приводим массивы к одинаковой длине
            min_len = min(len(self.prices), len(self.ema_values)) if self.ema_values else len(self.prices)
            self.prices = self.prices[:min_len]
            self.ema_values = self.ema_values[:min_len] if self.ema_values else self.prices.copy()
            print(f"Массивы приведены к длине: {min_len}")
        
        for i in range(len(self.prices)):
            if i % 100 == 0:  # Отладочное сообщение каждые 100 точек
                print(f"Обработка точки {i}/{len(self.prices)}, цена: {self.prices[i]}")
            
            # Получаем значения как скаляры, а не как массивы
            price = float(self.prices[i]) if isinstance(self.prices[i], np.ndarray) else self.prices[i]
            ema = float(self.ema_values[i]) if isinstance(self.ema_values[i], np.ndarray) else self.ema_values[i]
            self.grid_center = ema

            new_direction = self.direction
            if i < len(self.features_df):
                try:
                    scaled = self.scaler.transform([self.features_df.iloc[i]])
                    predicted_trend = self.model.predict(scaled)[0]
                    new_direction = {1: "long", -1: "short", 0: "neutral"}[predicted_trend]
                    if new_direction != self.direction:
                        self.direction = new_direction
                        self.positions = []
                except Exception as e:
                    print(f"Ошибка при предсказании тренда: {e}")
            
            self.grid = self.generate_grid(self.grid_center)
            
            for order_type, order_price in self.grid:
                if (order_type == "buy" and price <= order_price) or (order_type == "sell" and price >= order_price):
                    volume = self.volume_by_direction(order_type)
                    self.positions.append(Position(order_type, order_price, volume))
            
            new_positions = []
            realized_profit = 0
            for pos in self.positions:
                target_exit_price = self.find_target_exit_price(pos, price)
                if target_exit_price is not None and (
                    (pos.order_type == "buy" and price >= target_exit_price) or
                    (pos.order_type == "sell" and price <= target_exit_price)
                ):
                    profit = pos.close(target_exit_price)
                    self.log_order(pos.order_type, pos.entry_price, pos.volume, target_exit_price, profit)
                    realized_profit += profit
                else:
                    new_positions.append(pos)
            self.positions = [p for p in new_positions if not p.closed]
            self.balance += realized_profit
            
            floating = sum(p.floating_profit(price) for p in self.positions)
            equity = self.balance + floating
            drawdown = self.get_drawdown(equity)
            
            if drawdown >= 0.05 and self.hedge_position is None:
                self.open_hedge(price)
            
            if self.hedge_position:
                combined_pnl = self.floating_pnl(price)
                if combined_pnl >= 0 or new_direction != self.direction:
                    profit = self.hedge_position.close(price)
                    self.log_hedge(self.hedge_position.order_type, self.hedge_position.entry_price, self.hedge_position.volume, price, profit)
                    self.balance += profit
                    self.hedge_history.append(self.hedge_position)
                    self.hedge_position = None
            
            if equity > self.max_equity:
                self.max_equity = equity
            if equity < self.max_equity * (1 - self.direction_change_threshold):
                self.switch_direction()
            
            self.equity_history.append(equity)
            self.balance_history.append(self.balance)
            self.direction_history.append(self.direction)
        
        print(f"Симуляция завершена. Длина equity_history: {len(self.equity_history)}")
        if not self.equity_history:
            print("ВНИМАНИЕ: equity_history пуст!")

    def simulate_with_ticks(self, tick_data, batch_size=10000):
        """
        Симуляция с использованием тиковых данных, обработка порциями.
        """
        total_ticks = len(tick_data)
        for batch_start in range(0, total_ticks, batch_size):
            batch_end = min(batch_start + batch_size, total_ticks)
            batch = tick_data.iloc[batch_start:batch_end]

            for i, row in batch.iterrows():
                price = row['price']
                timestamp = row['timestamp']
                # Обработка текущего тика
                # ...existing tick processing logic...

                # Логирование прогресса
                if i % (total_ticks // 100) == 0:  # Каждые 1%
                    progress = (i / total_ticks) * 100
                    print(f"Прогресс симуляции: {progress:.1f}%")

            # Обновление графика после каждой порции
            self.equity_history.append(self.balance + sum(p.floating_profit(price) for p in self.positions))

        print("Симуляция завершена.")

    def switch_direction(self):
        if self.direction == "neutral":
            self.direction = "short"
        elif self.direction == "short":
            self.direction = "long"
        else:
            self.direction = "neutral"
        self.positions = []

    def plot(self):
        """
        Построение графика на основе текущих данных симуляции.
        """
        if not self.equity_history or not self.balance_history:
            print("Нет данных для построения графика.")
            return

        x = pd.date_range(start="2024-01-01", periods=len(self.equity_history), freq="s")
        plt.figure(figsize=(14, 6))
        plt.plot(x, self.equity_history, label="Equity")
        plt.plot(x, self.balance_history, linestyle="--", label="Balance")
        plt.title("Grid Strategy with ML & Hedge")
        plt.xlabel("Time")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Вывод итогового резюме симуляции
        self.print_simulation_summary()

    def print_simulation_summary(self):
        """
        Вывод итогового резюме симуляции.
        """
        final_equity = self.equity_history[-1] if self.equity_history else self.initial_balance
        max_drawdown = self.calculate_max_drawdown() * 100
        total_commission = self.calculate_commission()

        print("\n=== Simulation Summary ===")
        print(f"Final Equity: {final_equity:.2f} USD")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Commission Paid: {total_commission:.2f} USD")
        print("==========================")

        # Сохранение итогов в CSV
        summary_data = {
            "Final Equity": [final_equity],
            "Max Drawdown (%)": [max_drawdown],
            "Total Commission Paid (USD)": [total_commission]
        }
        pd.DataFrame(summary_data).to_csv("simulation_summary.csv", index=False)
        print("Simulation summary saved to simulation_summary.csv")

        # Сохранение детализированного резюме
        detailed_data = {
            "Equity History": self.equity_history,
            "Balance History": self.balance_history,
            "Direction History": self.direction_history
        }
        pd.DataFrame(detailed_data).to_csv("detailed_simulation_summary.csv", index=False)
        print("Detailed simulation summary saved to detailed_simulation_summary.csv")

    def get_drawdown(self, equity):
        if not self.balance_history:
            return 0.0
        current_balance = self.balance_history[-1]
        drawdown = (current_balance - equity) / current_balance if current_balance > equity else 0.0
        return drawdown

    def floating_pnl(self, price):
        grid_pnl = sum(p.floating_profit(price) for p in self.positions)
        hedge_pnl = self.hedge_position.floating_profit(price) if self.hedge_position else 0
        return grid_pnl + hedge_pnl

    def open_hedge(self, price):
        direction = self.detect_grid_bias()
        if direction is None:
            direction = self.direction
        if direction == "neutral":
            return
        hedge_type = "sell" if direction == "buy" else "buy"
        volume = (self.balance * 0.5) / price
        self.hedge_position = Position(hedge_type, price, volume)
        self.log_hedge(hedge_type, price, volume)

    def detect_grid_bias(self):
        buys = sum(1 for p in self.positions if p.order_type == "buy")
        sells = sum(1 for p in self.positions if p.order_type == "sell")
        if buys > sells:
            return "buy"
        elif sells > buys:
            return "sell"
        else:
            return None

    def calculate_max_drawdown(self):
        max_drawdown = 0
        peak = self.equity_history[0]
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_commission(self):
        total_commission = 0
        for pos in self.positions + self.hedge_history:
            entry_commission = pos.entry_price * pos.volume * 0.0002
            exit_commission = (pos.exit_price or 0) * pos.volume * 0.0004
            total_commission += entry_commission + exit_commission
        for log in getattr(self, 'order_log', []):
            entry_commission = log['Entry Price'] * log['Volume'] * 0.0002
            exit_commission = (log['Exit Price'] or 0) * log['Volume'] * 0.0004
            total_commission += entry_commission + exit_commission
        return total_commission
