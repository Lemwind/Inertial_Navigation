# test_kf_1d.py
import numpy as np
import matplotlib.pyplot as plt
from Kalman import Kalman

# 参数
N = 2000                       # 采样点数
DT= 0.001                      # 1 ms

# 实例化滤波器
kf = Kalman(P0=1.0, Q_accel_noise=0.02, R_pos_noise=0.5)

# 存储数组
true_pos = np.zeros(N)
true_vel = np.zeros(N)
kf_pos   = np.zeros(N)
kf_vel   = np.zeros(N)
obs_pos  = np.zeros(N)

# 生成“真实”轨迹
for k in range(1, N):
    true_vel[k] = true_vel[k-1] + 0.05*(np.random.rand()-0.3)
    if true_vel[k] < 0.01:          # 保证 >0
        true_vel[k] = 0.01
    true_pos[k] = true_pos[k-1] + true_vel[k]*DT

# 滤波主循环
for k in range(N):
    # 模拟加速度计读数
    if k == 0:
        accel = 0.0
    else:
        accel = (true_vel[k] - true_vel[k-1])/DT + 0.5*(np.random.rand()-0.5)

    # 模拟带噪声的位置观测
    obs_z = true_pos[k] + 0.7*np.random.randn()

    # 1. 预测
    kf.predict(accel)
    # 2. 更新
    kf.update(obs_z)

    # 记录
    kf_pos[k]  = kf.pos
    kf_vel[k]  = kf.vel
    obs_pos[k] = obs_z

# 绘图
t = np.arange(N)*DT
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(t, true_pos, label='true pos')
plt.plot(t, obs_pos,  '.', markersize=2, label='observed pos')
plt.plot(t, kf_pos,   label='KF pos')
plt.legend(); plt.xlabel('time (s)'); plt.ylabel('position (m)')

plt.subplot(1,2,2)
plt.plot(t, true_vel, label='true vel')
plt.plot(t, kf_vel,   label='KF vel')
plt.legend(); plt.xlabel('time (s)'); plt.ylabel('velocity (m/s)')
plt.tight_layout()
plt.show()