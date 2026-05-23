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

    def add_training_result(self, training_number, accumulated_reward, sample_count, metrics=None, **extra_metrics):
        """Append one training-update record.

        Backwards compatible with the original signature; additionally accepts:
        - metrics: optional dict of scalar values
        - **extra_metrics: additional scalar key/value pairs
        """

        row = {
            "training_number": int(training_number),
            "accumulated_reward": float(accumulated_reward),
            "sample_count": int(sample_count),
        }

        if metrics:
            if not isinstance(metrics, dict):
                raise TypeError("metrics must be a dict if provided")
            row.update(metrics)

        if extra_metrics:
            row.update(extra_metrics)

        self.rows.append(row)

    def export_csv(self):
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w", newline="") as csvfile:
            base = ["training_number", "accumulated_reward", "sample_count"]
            extra_keys = []
            if self.rows:
                keys = set()
                for r in self.rows:
                    keys.update(r.keys())
                extra_keys = sorted(k for k in keys if k not in base)

            fieldnames = base + extra_keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"[Export] Training rewards exported to {self.file_path}")
