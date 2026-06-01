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

        sample_count_i = int(sample_count)
        accumulated_reward_f = float(accumulated_reward)
        reward_per_sample = (
            accumulated_reward_f / sample_count_i
            if sample_count_i > 0
            else float("nan")
        )

        row = {
            "training_number": int(training_number),
            "accumulated_reward": accumulated_reward_f,
            "sample_count": sample_count_i,
            "reward_per_sample": float(reward_per_sample),
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
            base = ["training_number", "accumulated_reward", "sample_count", "reward_per_sample"]
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
