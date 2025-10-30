import numpy as np
from Allan import allan_variance, plot_allan

def simulate_gyro(fs: float, duration: float, arw: float = 0.05, rrw: float = 0.003, bias: float = 1.0):
    """
    生成含角度随机游走 + 速率随机游走 + 常值偏置的陀螺零偏数据
    单位：角速率 °/h
    """
    n = int(fs * duration)
    dt = 1 / fs
    # 白噪声 → 角度随机游走
    w = np.random.randn(n) * arw / np.sqrt(dt)
    # 随机游走 → 速率随机游走
    r = np.cumsum(np.random.randn(n) * rrw * np.sqrt(dt))
    return bias + w + r


if __name__ == '__main__':
    fs = 100            # Hz
    T = 2 * 3600        # 采集 2 h
    data = simulate_gyro(fs, T, arw=0.05, rrw=0.003, bias=1.0)

    tau, adev = allan_variance(data, fs, tau_min=0.1, tau_max=500)
    plot_allan(tau, adev, unit='°/h')