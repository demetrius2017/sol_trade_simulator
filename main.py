
import pandas as pd
from grid_simulator_with_ml_hedged import SimpleGridSimulator, Position

# Загрузка подготовленного файла
df = pd.read_csv("solana_minute_compact_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
prices = df["price"].values
timestamps = df["timestamp"].values

# Оптимизированный симулятор с простым find_target_exit_price
class OptimizedGridSimulator(SimpleGridSimulator):
    def find_target_exit_price(self, position, current_price):
        step_price = self.grid_step * position.entry_price
        if position.order_type == "buy":
            return position.entry_price + step_price
        else:
            return position.entry_price - step_price

    def simulate(self):
        print(f"Запуск: {len(self.prices)} точек")
        if isinstance(self.prices, list) is False:
            self.prices = self.prices.tolist()
        if not self.ema_values:
            self.ema_values = self.prices.copy()

        for i in range(len(self.prices)):
            price = self.prices[i]
            if self.grid_center is None:
                self.grid_center = price
            elif abs(price - self.grid_center) / self.grid_center > self.grid_step * self.grid_size:
                self.grid_center = price

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
            self.equity_history.append(equity)
            self.balance_history.append(self.balance)

        print(f"Симуляция завершена. Точек: {len(self.equity_history)}")

# Запуск симуляции
sim = OptimizedGridSimulator(
    prices=prices,
    ema_values=[],
    features_df=pd.DataFrame(),
    model=None,
    scaler=None,
    timestamps=timestamps
)

sim.simulate()
sim.plot()
sim.print_simulation_summary()
