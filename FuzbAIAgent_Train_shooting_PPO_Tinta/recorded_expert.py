"""Table recordings -> calibrated, uniformly sampled expert transitions."""
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from reward.single_bar_shoting import player_alignment_target

OBS_COLUMNS = (
    'ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'ball_rod_dist_x',
    'target_rod_error', 'rod_position', 'rod_angle',
)


def observation(ball_x, ball_y, ball_vx, ball_vy, rod_position, rod_angle, rod_info):
    """Simulator camera units: mm, m/s, translation [0,1], angle pi/32."""
    _, target, _ = player_alignment_target(ball_y=ball_y, rod_info=rod_info)
    return np.array([
        (ball_x - 605) / 605, (ball_y - 350) / 350,
        np.clip(ball_vx / 5, -2, 2), np.clip(ball_vy / 5, -2, 2),
        (ball_x - rod_info['position']) / 605, target - rod_position,
        rod_position, np.clip(rod_angle / 32, -1, 1),
    ], dtype=np.float32)


@dataclass
class RecordingConfig:
    # These defaults are assumptions, not an inferred table calibration.
    sample_dt: float = 0.05
    max_gap: float = 0.1
    rod_index: int = 3
    angle_units: str = 'radians'
    angle_sign: float = 1.0
    angle_offset_radians: float = 0.0
    x_scale: float = 1.0
    x_offset_mm: float = 0.0
    y_scale: float = 1.0
    y_offset_mm: float = 0.0
    velocity_scale: float = 1.0
    invert_rod_position: bool = False
    rotate_table_180: bool = False

    def __post_init__(self):
        if self.sample_dt <= 0 or self.max_gap < self.sample_dt:
            raise ValueError('Require 0 < sample_dt <= max_gap')
        if not 0 <= self.rod_index <= 7:
            raise ValueError('rod_index must be in 0..7')
        if self.angle_units not in {'radians', 'degrees', 'simulator'}:
            raise ValueError('angle_units must be radians, degrees, or simulator')
        if not all(np.isfinite(v) for v in asdict(self).values() if isinstance(v, (float, int))):
            raise ValueError('Calibration must contain finite numbers')
        if self.x_scale == 0 or self.y_scale == 0 or self.angle_sign == 0:
            raise ValueError('Coordinate and angle scales cannot be zero')

    def angle(self, value):
        factors = {'radians': 1.0, 'degrees': np.pi / 180, 'simulator': np.pi / 32}
        return float(value) * factors[self.angle_units] * self.angle_sign + self.angle_offset_radians


class ExpertTransitionDataset:
    """Each file is an episode; invalid rows and time gaps split it into segments.

    Transitions never cross a segment. Only segments starting at the first valid
    row of a file supply reset states; tracking reacquisition is not an episode
    start. Metadata and a validation report make calibration reproducible.
    """
    def __init__(self, path, geometry, config=None):
        self.config = config or RecordingConfig()
        self.rod_info = next(r for r in geometry['rods'] if r['id'] == 4)
        source = Path(path)
        files = sorted(source.glob('*.csv')) if source.is_dir() else [source]
        if not files or not all(f.is_file() for f in files):
            raise ValueError(f'No recording CSV files found at {path}')
        self.episodes = []
        self.starts = []
        report = dict(files=len(files), rows=0, invalid_rows=0, time_breaks=0, segments=0)
        cfg = self.config
        required = {'Timestamp', 'Ball_x', 'Ball_y', 'Ball_vx', 'Ball_vy', 'Ball_size'}
        required |= {f'Rod{i}_{suffix}' for i in range(8) for suffix in ('p', 'fi')}
        for file in files:
            segments, segment = [], []
            first_valid = None
            with file.open(newline='', encoding='utf-8-sig') as stream:
                reader = csv.DictReader(stream, delimiter=';')
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f'{file}: missing columns {sorted(missing)}')
                for row in reader:
                    report['rows'] += 1
                    try:
                        t = datetime.strptime(row['Timestamp'], '%d.%m.%Y %H:%M:%S.%f')
                        raw_x, raw_y = float(row['Ball_x']), float(row['Ball_y'])
                        positions = [float(row[f'Rod{i}_p']) for i in range(8)]
                        angles = [cfg.angle(row[f'Rod{i}_fi']) for i in range(8)]
                        values = np.array([
                            raw_x * cfg.x_scale + cfg.x_offset_mm,
                            raw_y * cfg.y_scale + cfg.y_offset_mm,
                            float(row['Ball_vx']) * cfg.velocity_scale * cfg.x_scale,
                            float(row['Ball_vy']) * cfg.velocity_scale * cfg.y_scale,
                            *positions, *angles,
                        ], dtype=float)
                        if (not np.all(np.isfinite(values)) or raw_x < 0 or raw_y < 0
                                or not float(row['Ball_size']) > 0
                                or not 0 <= values[0] <= 1210 or not 0 <= values[1] <= 700
                                or any(not -0.02 <= v <= 1.02 for v in positions)):
                            raise ValueError('invalid observation')
                        values[4:12] = np.clip(values[4:12], 0, 1)
                        if cfg.invert_rod_position:
                            values[4:12] = 1 - values[4:12]
                        if cfg.rotate_table_180:
                            values[0:2] = [1210 - values[0], 700 - values[1]]
                            values[2:4] *= -1
                            values[4:12] = 1 - values[4:12][::-1]
                            values[12:20] = -values[12:20][::-1]
                    except (ValueError, TypeError, KeyError):
                        report['invalid_rows'] += 1
                        if segment:
                            segments.append(segment)
                        segment = []
                        continue
                    if first_valid is None:
                        first_valid = t
                    if segment and not 0 < (t - segment[-1][0]).total_seconds() <= cfg.max_gap:
                        report['time_breaks'] += 1
                        segments.append(segment)
                        segment = []
                    segment.append((t, values))
            if segment:
                segments.append(segment)
            for segment in segments:
                times = np.array([(r[0] - segment[0][0]).total_seconds() for r in segment])
                if times[-1] + 1e-9 < cfg.sample_dt:
                    continue
                raw = np.stack([r[1] for r in segment])
                # Interpolate unwrapped physical angles, never across time gaps.
                raw[:, 12:] = np.unwrap(raw[:, 12:], axis=0)
                grid = np.arange(int((times[-1] + 1e-7) / cfg.sample_dt) + 1) * cfg.sample_dt
                sampled = np.stack([np.interp(grid, times, raw[:, i]) for i in range(20)], axis=1)
                # Map recorded active rod to physical red midfield. Other rods
                # retain their table index (full remapping requires calibration).
                active_index = 7 - cfg.rod_index if cfg.rotate_table_180 else cfg.rod_index
                if active_index != 3:
                    sampled[:, [7, 4 + active_index]] = sampled[:, [4 + active_index, 7]]
                    sampled[:, [15, 12 + active_index]] = sampled[:, [12 + active_index, 15]]
                states = np.stack([observation(*v[:4], v[7], v[15] * 32 / np.pi, self.rod_info) for v in sampled])
                self.episodes.append(states)
                report['segments'] += 1
                if segment[0][0] == first_valid:
                    self.starts.append({'state': sampled[0].tolist(), 'duration': float(grid[-1]), 'file': file.name})
        if not self.episodes or not self.starts:
            raise ValueError('No valid transitions/episode starts after preprocessing; check calibration and gaps')
        transitions = np.concatenate([np.concatenate([s[:-1], s[1:]], axis=1) for s in self.episodes])
        self.transitions = torch.as_tensor(transitions, dtype=torch.float32)
        self.report = {**report, 'transitions': len(transitions), 'reset_states': len(self.starts), 'config': asdict(cfg)}
        print('[ExpertDataset] ' + json.dumps(self.report))

    def sample(self, batch_size, device):
        indices = torch.randint(len(self.transitions), (int(batch_size),))
        return self.transitions[indices].to(device)
