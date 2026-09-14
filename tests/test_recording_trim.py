import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'FuzbAIAgent_Train_shooting_PPO_Tinta'
sys.path.insert(0, str(ROOT))
from recorded_expert import RecordingConfig, ExpertTransitionDataset


class TrimmingTests(unittest.TestCase):
    def dataset(self, config, invalid_middle=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.csv'
            rows = []
            for i in range(21):
                row = dict(Timestamp=(datetime(2026, 1, 1) + timedelta(seconds=i*.05)).strftime('%d.%m.%Y %H:%M:%S.%f'),
                    Ball_x=500+i, Ball_y=350, Ball_vx=.1, Ball_vy=0,
                    Ball_size=0 if invalid_middle and i == 10 else 34)
                row.update({f'Rod{j}_{suffix}': .5 if suffix == 'p' else 0 for j in range(8) for suffix in ('p', 'fi')})
                rows.append(row)
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=';')
                writer.writeheader(); writer.writerows(rows)
            original = path.read_bytes()
            result = ExpertTransitionDataset(path, json.loads((ROOT/'geometry.json').read_text()), config)
            self.assertEqual(path.read_bytes(), original)
            return result

    def test_middle_half_and_new_spawn(self):
        d = self.dataset(RecordingConfig(trim_start_fraction=.25, trim_end_fraction=.25))
        self.assertEqual(len(d.transitions), 10)
        self.assertEqual(d.starts[0]['state'][0], 505)
        self.assertAlmostEqual(d.starts[0]['duration'], .5)
        self.assertEqual(d.report['trimmed_rows'], 10)

    def test_gap_is_not_bridged_or_trimmed_again(self):
        d = self.dataset(RecordingConfig(trim_start_fraction=.25, trim_end_fraction=.25), True)
        self.assertEqual(len(d.transitions), 8)
        self.assertEqual(len(d.episodes), 2)
        self.assertEqual(len(d.starts), 1)
        self.assertAlmostEqual(d.starts[0]['duration'], .2)

    def test_legacy_configuration_keeps_full_recording(self):
        self.assertEqual(len(self.dataset(RecordingConfig()).transitions), 20)

    def test_invalid_fractions(self):
        for start, end in [(-.1, 0), (.5, .5), (float('nan'), 0), (0, 1.1)]:
            with self.assertRaises(ValueError):
                RecordingConfig(trim_start_fraction=start, trim_end_fraction=end)
