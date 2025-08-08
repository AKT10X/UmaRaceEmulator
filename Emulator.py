from UmaFrameProcessor import UmaState  # 定義ファイル名に合わせてください

# 例: シミュレーションを実行し、結果を取得
uma = UmaState(initial_d=0.0, lane=1.0, initial_v=20.2, acceleration=0.0, initial_rho=rho0)
frames = uma.run_until_finish(distances, xs, zs, lane_width=11.25, dt=0.0666, rho0=rho0)

# フレームごとに出力例
for i, (r, rho, d, v) in enumerate(frames):
    print(f"Frame {i}: pos={r}, rho={rho:.3f}, d={d:.2f}, v={v:.2f}")
