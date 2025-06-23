import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 定数設定 ---
dt = 0.0666           # 1フレームあたりの時間[s]
v = 20.2              # 常速[m/s]
lane_width = 11.25    # レーン幅[m]
skip = 5              # フレーム間引き

# --- キーフレーム読み込み ---
df = pd.read_csv('コース対座標 のコピー - シート6.csv', header=None, encoding='utf-8')
distances = df.iloc[:, 0].values
xs = df.iloc[:, 1].values
zs = df.iloc[:, 2].values

# --- シミュレーション関数 ---
def simulate(lane):
    d = distances[0]
    prev_r = None
    d_prev = None
    positions = []
    r0 = np.array([xs[0], zs[0]])
    r1 = np.array([xs[1], zs[1]])
    rho0 = np.linalg.norm(r1 - r0) / (distances[1] - distances[0])
    while d < distances[-1]:
        idx = np.searchsorted(distances, d) - 1
        idx = max(0, min(idx, len(distances)-2))
        d_i, x_i, z_i = distances[idx], xs[idx], zs[idx]
        d_ip1, x_ip1, z_ip1 = distances[idx+1], xs[idx+1], zs[idx+1]
        t = (d - d_i) / (d_ip1 - d_i)
        base = np.array([x_i, z_i]) * (1 - t) + np.array([x_ip1, z_ip1]) * t
        seg = np.array([x_ip1 - x_i, z_ip1 - z_i])
        seg_unit = seg / np.linalg.norm(seg)
        normal = np.array([ seg_unit[1], -seg_unit[0]])
        r = base + normal * (lane * lane_width)
        if prev_r is not None:
            delta_r = np.linalg.norm(r - prev_r)
            rho = delta_r / (d - d_prev)
        else:
            rho = rho0
        positions.append(r)
        d_prev = d
        prev_r = r
        d += (v * dt) / max(1.0, rho / rho0)
    return np.array(positions)

# 両者シミュレーション
pos1 = simulate(1.0)[::skip]
pos05 = simulate(0.5)[::skip]
dt_ss = dt * skip
n_frames = min(len(pos1), len(pos05))

# --- リアルタイム同期アニメーション ---
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(xs, zs, color='gray', linewidth=1)
horse1, = ax.plot([], [], 'ro', label='Lane 1')
horse2, = ax.plot([], [], 'bo', label='Lane 0.5')
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.set_aspect('equal')
ax.legend()
fig.show()

# --- アニメーションループ ---
for i in range(n_frames):
    # ← ここを修正 ↓
    horse1.set_data([pos1[i,0]],   [pos1[i,1]])
    horse2.set_data([pos05[i,0]],  [pos05[i,1]])
    fig.canvas.draw()
    plt.pause(dt_ss)

plt.ioff()
plt.show()

plt.ioff()
