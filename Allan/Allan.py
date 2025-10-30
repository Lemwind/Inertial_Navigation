import numpy as np

def allan_variance(x: np.ndarray, fs: float, tau_min: float = 0.01, tau_max: float = 1000):
    """
    非重叠 Allan 方差 / Allan 偏差
    参数
    ----
    x      : 1-D array，原始零偏序列（任意单位）
    fs     : 采样频率 Hz
    tau_min, tau_max : 感兴趣的 τ 范围（秒）

    返回
    ----
    tau   : 1-D array，平均时间 τ（s）
    adev  : 1-D array，Allan 标准差 σ(τ)（与 x 同单位）
    """
    n = x.size
    dt = 1 / fs
    m_max = n // 2
    # 指数间隔选点，保证 log-log 图均匀
    m_all = np.unique(np.logspace(0, np.log10(m_max), 300, dtype=int))
    m_all = m_all[(m_all >= 1) & (m_all <= m_max)]
    tau_all = m_all * dt
    mask = (tau_all >= tau_min) & (tau_all <= tau_max)
    m_all, tau_all = m_all[mask], tau_all[mask]

    adev = np.empty_like(tau_all)
    for i, m in enumerate(m_all):
        k = n // m
        # 分段平均
        theta = np.cumsum(x, dtype=float)        # 积分角（若 x 为角速率）
        theta = theta[:k*m:m] - theta[m-1::m]    # 每箱积分和
        diff = np.diff(theta)
        avar = np.sum(diff**2) / (2 * (k - 1) * m**2)
        adev[i] = np.sqrt(avar)
    return tau_all, adev


def plot_allan(tau, adev, unit='°/h'):
    import matplotlib.pyplot as plt
    plt.loglog(tau, adev, '-o', markersize=2)
    plt.xlabel('τ (s)'); plt.ylabel(f'Allan deviation ({unit})')
    plt.grid(True, which='both', ls='--', lw=0.4)
    plt.tight_layout()
    plt.show()