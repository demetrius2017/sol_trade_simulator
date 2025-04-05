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

    def find_target_exit_price(self, position):
        """
        Определяет целевую цену закрытия для позиции на основе сетки.
        """
        if position.order_type == "buy":
            # Найти ближайший sell-уровень выше entry_price
            target_prices = [price for order_type, price in self.grid if order_type == "sell" and price > position.entry_price]
        else:
            # Найти ближайший buy-уровень ниже entry_price
            target_prices = [price for order_type, price in self.grid if order_type == "buy" and price < position.entry_price]

        return min(target_prices, default=None) if position.order_type == "buy" else max(target_prices, default=None)

    def simulate(self):
        for i in range(len(self.prices)):
            price = self.prices[i]
            ema = self.ema_values[i]
            self.grid_center = ema

            new_direction = self.direction
            if i < len(self.features_df):
                scaled = self.scaler.transform([self.features_df.iloc[i]])
                predicted_trend = self.model.predict(scaled)[0]
                new_direction = {1: "long", -1: "short", 0: "neutral"}[predicted_trend]
                if new_direction != self.direction:
                    self.direction = new_direction
                    self.positions = []

            self.grid = self.generate_grid(self.grid_center)

            for order_type, order_price in self.grid:
                if (order_type == "buy" and price <= order_price) or (order_type == "sell" and price >= order_price):
                    volume = self.volume_by_direction(order_type)
                    self.positions.append(Position(order_type, order_price, volume))

            new_positions = []
            realized_profit = 0
            for pos in self.positions:
                target_exit_price = self.find_target_exit_price(pos)
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

    def switch_direction(self):
        if self.direction == "neutral":
            self.direction = "short"
        elif self.direction == "short":
            self.direction = "long"
        else:
            self.direction = "neutral"
        self.positions = []

    def plot(self):
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
