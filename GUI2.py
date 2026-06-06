import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==============================
# Professional Resource Dashboard
# ==============================

class ResourceForecastApp:

    def __init__(self, root):
        self.root = root
        self.root.title("IT Resource Forecasting & Optimization Dashboard")
        self.root.geometry("1100x700")
        self.root.configure(bg="white")

        self.df = None
        self.model = None
        self.canvas = None

        self.build_ui()

    # ------------------------------
    # UI Layout
    # ------------------------------
    def build_ui(self):

        title = tk.Label(
            self.root,
            text="🚀 IT Resource Demand Forecasting System",
            font=("Segoe UI", 20, "bold"),
            bg="white",
            fg="#222"
        )
        title.pack(pady=15)

        subtitle = tk.Label(
            self.root,
            text="Forecast CPU Demand • Optimize Allocation • Prevent Bottlenecks",
            font=("Segoe UI", 11),
            bg="white",
            fg="gray"
        )
        subtitle.pack()

        # Main Frame
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill="both", expand=True, pady=15)

        # Left Panel
        left_panel = tk.Frame(main_frame, bg="white", width=400)
        left_panel.pack(side="left", fill="y", padx=20)

        # Right Panel (Graph)
        self.right_panel = tk.Frame(main_frame, bg="white")
        self.right_panel.pack(side="right", fill="both", expand=True)

        # Buttons
        ttk.Button(
            left_panel,
            text="📂 Load Dataset (CSV)",
            command=self.load_csv
        ).pack(fill="x", pady=10)

        ttk.Button(
            left_panel,
            text="🤖 Train Model + Forecast",
            command=self.train_forecast
        ).pack(fill="x", pady=10)

        ttk.Button(
            left_panel,
            text="🧹 Clear Output",
            command=self.clear_output
        ).pack(fill="x", pady=10)

        # Output Box
        self.output_box = tk.Text(
            left_panel,
            height=25,
            font=("Consolas", 10),
            wrap="word"
        )
        self.output_box.pack(fill="both", expand=True, pady=15)

        self.output_box.insert("end", "📌 Load a dataset to begin...\n")

    # ------------------------------
    # Load Dataset
    # ------------------------------
    def load_csv(self):

        file_path = filedialog.askopenfilename(
            title="Select Resource Usage Dataset",
            filetypes=[("CSV Files", "*.csv")]
        )

        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df = self.df.sort_values("timestamp")

            messagebox.showinfo("Success", "Dataset Loaded Successfully!")

            self.output_box.insert("end", "\n✅ Dataset Loaded Successfully!\n")
            self.output_box.insert("end", str(self.df.head()) + "\n")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")

    # ------------------------------
    # Train + Forecast
    # ------------------------------
    def train_forecast(self):

        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return

        try:
            df = self.df.copy()

            # Feature Engineering
            df["hour"] = df["timestamp"].dt.hour
            df["day"] = df["timestamp"].dt.day
            df["weekday"] = df["timestamp"].dt.weekday

            features = [
                "hour", "day", "weekday",
                "memory_utilization",
                "storage_utilization",
                "network_bandwidth"
            ]

            X = df[features]
            y = df["cpu_utilization"]

            # Train-Test Split
            split_index = int(len(df) * 0.8)

            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Train Model
            self.model = RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            # Predict
            y_pred = self.model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.output_box.insert("end", "\n==============================\n")
            self.output_box.insert("end", "📊 Model Performance\n")
            self.output_box.insert("end", f"MAE  = {mae:.2f}\n")
            self.output_box.insert("end", f"RMSE = {rmse:.2f}\n")

            # Scaling Decisions
            self.output_box.insert("end", "\n⚡ Resource Allocation Decisions\n")

            for i in range(5):
                cpu = y_pred[i]

                if cpu > 80:
                    decision = "Scale UP 🚀"
                elif cpu < 30:
                    decision = "Scale DOWN ⬇"
                else:
                    decision = "Normal ✅"

                self.output_box.insert(
                    "end",
                    f"Prediction {i+1}: CPU={cpu:.2f}% → {decision}\n"
                )

            # Plot Forecast
            self.plot_graph(y_test, y_pred)

            messagebox.showinfo("Done", "Forecasting Completed Successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Training Failed:\n{e}")

    # ------------------------------
    # Plot Graph in GUI
    # ------------------------------
    def plot_graph(self, actual, predicted):

        # Clear old graph
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig = plt.Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(actual.values[:50], label="Actual CPU")
        ax.plot(predicted[:50], label="Predicted CPU")

        ax.set_title("CPU Demand Forecast")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("CPU Utilization (%)")
        ax.legend()

        self.canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=30)

    # ------------------------------
    # Clear Output
    # ------------------------------
    def clear_output(self):
        self.output_box.delete("1.0", "end")
        self.output_box.insert("end", "🧹 Output cleared.\n")


# ==============================
# Run Application
# ==============================
if __name__ == "__main__":

    root = tk.Tk()

    # Modern ttk theme
    style = ttk.Style()
    style.theme_use("clam")

    app = ResourceForecastApp(root)

    root.mainloop()
