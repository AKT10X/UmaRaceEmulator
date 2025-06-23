import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
DT = 0.0666  # seconds per frame (approx 15fps)
LANE_WIDTH = 11.25
TRACK_CSV = 'コース対座標 のコピー - シート6.csv'

# Load track keyframes
track_df = pd.read_csv(TRACK_CSV, header=None, encoding='utf-8')
DISTANCES = track_df.iloc[:, 0].values
XS = track_df.iloc[:, 1].values
ZS = track_df.iloc[:, 2].values


def interpolate_position(distance: float) -> np.ndarray:
    """Return (x, z) for given distance along track using linear interpolation."""
    idx = np.searchsorted(DISTANCES, distance) - 1
    idx = max(0, min(idx, len(DISTANCES) - 2))
    d0, d1 = DISTANCES[idx], DISTANCES[idx + 1]
    x0, x1 = XS[idx], XS[idx + 1]
    z0, z1 = ZS[idx], ZS[idx + 1]
    t = (distance - d0) / (d1 - d0)
    base = np.array([x0 * (1 - t) + x1 * t, z0 * (1 - t) + z1 * t])
    seg = np.array([x1 - x0, z1 - z0])
    seg_unit = seg / np.linalg.norm(seg)
    normal = np.array([seg_unit[1], -seg_unit[0]])
    return base, normal


class Horse:
    def __init__(self, lane: float, speed: float, color: str):
        self.distance = DISTANCES[0]
        self.speed = speed
        self.lane = lane
        self.color = color
        self.positions = []

    def step(self):
        base, normal = interpolate_position(self.distance)
        position = base + normal * (self.lane * LANE_WIDTH)
        self.positions.append(position)
        self.distance += self.speed * DT
        return position


def simulate(num_horses=3, duration=20):
    horses = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(num_horses):
        lane = (i + 1) / (num_horses + 1)
        speed = 19.0 + np.random.uniform(-0.5, 0.5)  # m/s
        horses.append(Horse(lane, speed, colors[i % len(colors)]))

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(XS, ZS, color='gray', linewidth=1)
    scatters = [ax.plot([], [], 'o', color=h.color)[0] for h in horses]
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('Uma Musume Race Emulator')
    fig.show()

    n_frames = int(duration / DT)
    for _ in range(n_frames):
        for scatter, horse in zip(scatters, horses):
            pos = horse.step()
            scatter.set_data([pos[0]], [pos[1]])
        fig.canvas.draw()
        plt.pause(DT)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    simulate()
