import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 定数設定 ---
dt = 0.0666
v_base = 20.2
v_target = 22.0
a = 0.3
accel_duration = 10.0
lane_width = 11.25
skip = 5          # 大幅間引き
speed_factor = 4  # 再生4倍速

# --- キーフレーム読み込み ---
csv_path = 'コース対座標.csv'
df = pd.read_csv(csv_path, header=None)
distances = df.iloc[:, 0].values
xs = df.iloc[:, 1].values
zs = df.iloc[:, 2].values

# --- 基準距離比 ---
rho0 = np.linalg.norm([xs[1]-xs[0], zs[1]-zs[0]]) / (distances[1] - distances[0])

# --- シミュレーション ---
def simulate(lanes):
    n = len(lanes)
    d = np.full(n, distances[0], dtype=float)
    v_curr = np.full(n, v_base, dtype=float)
    prev_r = [None] * n
    d_prev = np.zeros(n, dtype=float)

    pos_list = [[] for _ in lanes]
    speed_list = [[] for _ in lanes]
    rho_list_all = [[] for _ in lanes]
    t = 0.0
    times = []

    while np.all(d < distances[-1]):
        # 各レーンの位置計算
        r_list = []
        for i, lane in enumerate(lanes):
            idx = np.clip(np.searchsorted(distances, d[i]) - 1, 0, len(distances) - 2)
            di, dip1 = distances[idx], distances[idx+1]
            xi, xip1 = xs[idx], xs[idx+1]
            zi, zip1 = zs[idx], zs[idx+1]
            alpha = (d[i] - di) / (dip1 - di)
            base = np.array([xi, zi]) * (1 - alpha) + np.array([xip1, zip1]) * alpha
            seg = np.array([xip1 - xi, zip1 - zi])
            u = seg / np.linalg.norm(seg)
            normal = np.array([u[1], -u[0]])
            r = base + normal * (lane * lane_width)
            r_list.append(r)

        # 走行距離比（ρ）の計算
        rho_vals = []
        for i in range(n):
            if prev_r[i] is not None:
                rho_vals.append(
                    np.linalg.norm(r_list[i] - prev_r[i]) / (d[i] - d_prev[i])
                )
            else:
                rho_vals.append(rho0)

        

        # データ記録
        times.append(t)
        for i in range(n):
            pos_list[i].append(r_list[i])
            speed_list[i].append(v_curr[i])
            rho_list_all[i].append(rho_vals[i])

        # 次ステップへ
        for i in range(n):
            d_prev[i], prev_r[i] = d[i], r_list[i]
            d[i] += (v_curr[i] * dt) / max(1.0, rho_vals[i] / rho0)
        t += dt

    # 配列に変換して返す
    times = np.array(times)
    pos_arr = [np.array(p) for p in pos_list]
    speed_arr = [np.array(s) for s in speed_list]
    rho_arr = [np.array(r) for r in rho_list_all]
    return times, pos_arr, speed_arr, rho_arr

# --- シミュレーション実行 & GIF作成 ---
times, pos_list, speed_list, rho_list_all = simulate([1.0, 0.5])
pos1, pos05 = pos_list
speed1, speed05 = speed_list
rho1, rho05 = rho_list_all

# 間引き
times_s = times[::skip]
p1 = pos1[::skip]
p2 = pos05[::skip]
s1 = speed1[::skip]
s2 = speed05[::skip]
r1 = rho1[::skip]
r2 = rho05[::skip]

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(xs, zs, color='gray', linewidth=1)
h1, = ax.plot([], [], 'ro', label='Lane1')
h2, = ax.plot([], [], 'bo', label='Lane0.5')
text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
               va='top', ha='left', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7))
ax.set_aspect('equal')
ax.axis('off')

def init():
    h1.set_data([], [])
    h2.set_data([], [])
    text.set_text('')
    return h1, h2, text

def update(frame):
    x1, z1 = p1[frame]
    x2, z2 = p2[frame]
    h1.set_data([x1], [z1])
    h2.set_data([x2], [z2])
    txt = (f'Lane1: v={s1[frame]:.1f}m/s, ρ={r1[frame]:.3f}\n'
           f'Lane0.5: v={s2[frame]:.1f}m/s, ρ={r2[frame]:.3f}')
    text.set_text(txt)
    return h1, h2, text

ani = FuncAnimation(fig, update, frames=len(times_s),
                    init_func=init, blit=True)
fps = speed_factor / (dt * skip)
gif_path = 'race2_no_accel.gif'
ani.save(gif_path, writer=PillowWriter(fps=fps))

print(f"生成したGIFファイル: {gif_path}")
