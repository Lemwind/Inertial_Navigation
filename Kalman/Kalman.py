import numpy as np

class Kalman:
    """
    一维直线小车卡尔曼滤波器
    状态 [position, velocity]^T
    输入：加速度计读数 a (m/s^2)
    观测：外部给的位置 z (m)
    调用间隔须固定为 1 ms (DT=0.001 s)
    """

    DT = 0.001                  # 1 ms
    F  = np.array([[1, DT],
                   [0, 1]], dtype=float)
    B  = np.array([[0.5*DT**2],
                   [DT]], dtype=float)
    H  = np.array([[1, 0]], dtype=float)

    def __init__(self, P0=1.0, Q_accel_noise=0.02, R_pos_noise=0.5):
        """
        P0            : 初始估计协方差（标量，单位矩阵*P0）
        Q_accel_noise : 加速度过程噪声方差 (m^2/s^4)
        R_pos_noise   : 位置观测噪声方差 (m^2)
        """
        self.X = np.array([[0.0], [0.0]])          # [pos; vel]
        self.P = np.eye(2) * P0
        self.Q = Q_accel_noise * self.B @ self.B.T
        self.R = np.array([[R_pos_noise]])

    def predict(self, accel: float):
        """根据加速度计读数做预测"""
        self.X = self.F @ self.X + self.B * accel
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: float):
        """根据外部位置观测做更新"""
        y = z - (self.H @ self.X)[0,0]
        S = self.H @ self.P @ self.H.T + self.R
        K = (self.P @ self.H.T) / S[0,0]           # 2×1
        self.X += K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def pos(self):
        return float(self.X[0,0])

    @property
    def vel(self):
        return float(self.X[1,0])