
import numpy as np
import matplotlib.pyplot as plt
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

    def close(self, exit_price):
        self.exit_price = exit_price
        self.closed = True
        return self.floating_profit(exit_price)


class SimpleGridSimulator:
    def __init__(self, prices, ema_values, features_df, model, scaler,
                 initial_balance=100000, grid_step=0.01, grid_size=10,
                 direction_change_threshold=0.1):
        self.prices = prices
        self.ema_values = ema_values
        self.features_df = features_df
        self.model = model
        self.scaler = scaler

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

    def generate_grid(self, center_price):
        grid = []
        for i in range(1, self.grid_size + 1):
            delta = self.grid_step * i
            buy_price = center_price * (1 - delta)
            sell_price = center_price * (1 + delta)
            grid.append(("buy", buy_price))
            grid.append(("sell", sell_price))
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

            # ML прогноз тренда
            if i < len(self.features_df):
                scaled = self.scaler.transform([self.features_df.iloc[i]])
                predicted_trend = self.model.predict(scaled)[0]
                new_direction = {1: "long", -1: "short", 0: "neutral"}[predicted_trend]
                if new_direction != self.direction:
                    self.direction = new_direction
                    self.positions = []  # закрываем все при смене

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
            self.equity_history.append(equity)
            self.balance_history.append(self.balance)
            self.direction_history.append(self.direction)

            if equity < self.max_equity * (1 - self.direction_change_threshold):
                self.switch_direction()
                self.max_equity = equity
            elif equity > self.max_equity:
                self.max_equity = equity

    def switch_direction(self):
        if self.direction == "neutral":
            self.direction = "short"
        elif self.direction == "short":
            self.direction = "long"
        else:
            self.direction = "neutral"
        self.positions = []

    def plot(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.equity_history, label="Equity")
        plt.plot(self.balance_history, label="Balance", linestyle="--")
        plt.title("Grid Strategy Simulation with ML Trend Direction")
        plt.xlabel("Steps")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(True)
        plt.show()
