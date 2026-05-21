import csv
import os
from datetime import datetime


class Export:
    def __init__(self, file_path=None):
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"training_reward_log_{timestamp}.csv"

        self.file_path = file_path
        self.rows = []

    def add_training_result(self, training_number, accumulated_reward, sample_count):
        self.rows.append({
            "training_number": int(training_number),
            "accumulated_reward": float(accumulated_reward),
            "sample_count": int(sample_count),
        })

    def export_csv(self):
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w", newline="") as csvfile:
            fieldnames = ["training_number", "accumulated_reward", "sample_count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"[Export] Training rewards exported to {self.file_path}")
