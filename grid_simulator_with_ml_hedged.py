
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
        return net_profit


class SimpleGridSimulator:
    def __init__(self, prices, ema_values, features_df, model, scaler,
                 initial_balance=100000, grid_step=0.004, grid_size=30,
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
                if pos.order_type == "buy" and price >= self.grid_center:
                    realized_profit += pos.close(price)
                elif pos.order_type == "sell" and price <= self.grid_center:
                    realized_profit += pos.close(price)
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
                    self.balance += self.hedge_position.close(price)
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
        if self.max_equity == 0:
            return 0.0
        return (self.max_equity - equity) / self.max_equity

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

    def detect_grid_bias(self):
        buys = sum(1 for p in self.positions if p.order_type == "buy")
        sells = sum(1 for p in self.positions if p.order_type == "sell")
        if buys > sells:
            return "buy"
        elif sells > buys:
            return "sell"
        else:
            return None
