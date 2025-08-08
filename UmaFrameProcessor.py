import numpy as np

class UmaState:
    """
    ウマ娘の状態を表すクラス。

    Attributes:
        d (float): 走行距離 [m]
        lane (float): 走行レーン
        v (float): 現在速度 [m/s]
        a (float): 加速度 [m/s^2]
        prev_rho (float): 前フレームの距離比
        prev_r (np.ndarray): 前フレームの xz 座標
        d_prev (float): 前フレームの走行距離
    """
    def __init__(self, initial_d, lane, initial_v, acceleration, initial_rho=1.0):
        self.d = initial_d
        self.lane = lane
        self.v = initial_v
        self.a = acceleration
        self.prev_rho = initial_rho
        self.prev_r = None
        self.d_prev = initial_d

    def update_frame(self, distances, xs, zs, lane_width, dt, rho0):
        """
        1フレーム分の状態を更新するメソッド。

        Args:
            distances (np.ndarray): キーフレームの走行距離配列 (N,)
            xs (np.ndarray): キーフレームの x 座標配列 (N,)
            zs (np.ndarray): キーフレームの z 座標配列 (N,)
            lane_width (float): 1レーンあたりの幅 [m]
            dt (float): フレーム時間 [s]
            rho0 (float): 基準距離比

        Returns:
            Tuple[np.ndarray, float]: 更新後の座標 r と距離比 rho
        """
        # 前フレームの走行距離を記録
        self.d_prev = self.d

        # 線形補間でベース座標を計算
        idx = np.searchsorted(distances, self.d) - 1
        idx = np.clip(idx, 0, len(distances) - 2)
        d_i, d_ip1 = distances[idx], distances[idx+1]
        x_i, x_ip1 = xs[idx], xs[idx+1]
        z_i, z_ip1 = zs[idx], zs[idx+1]
        t = (self.d - d_i) / (d_ip1 - d_i)
        base = np.array([x_i, z_i]) * (1 - t) + np.array([x_ip1, z_ip1]) * t

        # セグメント方向の単位ベクトルと法線ベクトル（外側方向）
        seg = np.array([x_ip1 - x_i, z_ip1 - z_i])
        seg_u = seg / np.linalg.norm(seg)
        normal = np.array([seg_u[1], -seg_u[0]])

        # 実座標 r を算出
        r = base + normal * (self.lane * lane_width)

        # 距離比 rho を計算
        if self.prev_r is not None:
            delta = np.linalg.norm(r - self.prev_r)
            rho = delta / (self.d - self.d_prev)
        else:
            rho = rho0

        # 速度と走行距離を更新
        self.v += self.a * dt
        self.d += (self.v * dt) / max(1.0, rho / rho0)

        # 前フレーム情報を更新
        self.prev_r = r
        self.prev_rho = rho

        return r, rho

    def run_until_finish(self, distances, xs, zs, lane_width, dt, rho0):
        """
        走行距離がコース距離を越えるまでフレーム処理を続ける。

        Args:
            distances (np.ndarray): キーフレームの走行距離配列
            xs (np.ndarray): キーフレームの x 座標配列
            zs (np.ndarray): キーフレームの z 座標配列
            lane_width (float): 1レーンあたりの幅 [m]
            dt (float): フレーム時間 [s]
            rho0 (float): 基準距離比

        Returns:
            List[Tuple[np.ndarray, float, float, float]]: 
                各フレームの (r, rho, d, v) のリスト
        """
        frames = []
        max_d = distances[-1]
        while self.d <= max_d:
            r, rho = self.update_frame(distances, xs, zs, lane_width, dt, rho0)
            frames.append((r, rho, self.d, self.v))
        return frames
