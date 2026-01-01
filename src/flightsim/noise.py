"""
噪声模型模块
实现过程噪声（Dryden风场湍流、气动摄动）和量测噪声（IMU、GPS）
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ============================================================================
# 噪声配置
# ============================================================================

@dataclass
class NoiseConfig:
    """
    噪声配置参数
    
    所有强度参数范围为 [0, 1]，0表示无噪声，1表示最大噪声
    """
    # 过程噪声
    wind_intensity: float = 0.0       # 风场湍流强度 (0-1)
    aero_perturbation: float = 0.0    # 气动摄动强度 (0-1)
    
    # IMU量测噪声配置
    imu_noise: float = 0.0            # IMU噪声强度 (0-1)
    imu_noise_type: str = 'white'     # IMU噪声类型
    imu_flicker_prob: float = 0.1
    imu_flicker_scale: float = 5.0
    imu_drift_rate: float = 0.002
    imu_colored_alpha: float = 0.9
    imu_timevar_period: float = 100.0  # 时变周期
    imu_timevar_amp: float = 1.0       # 时变幅度
    
    # GPS量测噪声配置
    gps_noise: float = 0.0            # GPS噪声强度 (0-1)
    gps_noise_type: str = 'white'     # GPS噪声类型
    gps_flicker_prob: float = 0.1
    gps_flicker_scale: float = 5.0
    gps_drift_rate: float = 0.002
    gps_colored_alpha: float = 0.9
    gps_timevar_period: float = 100.0  # 时变周期
    gps_timevar_amp: float = 1.0       # 时变幅度
    
    # 兼容旧接口的通用噪声类型（已废弃，保留向后兼容）
    noise_type: str = 'white'
    drift_rate: float = 0.001
    flicker_prob: float = 0.05
    flicker_scale: float = 3.0
    colored_alpha: float = 0.9
    
    def has_process_noise(self) -> bool:
        """检查是否有过程噪声"""
        return self.wind_intensity > 0 or self.aero_perturbation > 0
    
    def has_measurement_noise(self) -> bool:
        """检查是否有量测噪声"""
        return self.imu_noise > 0 or self.gps_noise > 0



# ============================================================================
# Dryden 风场湍流模型
# ============================================================================

class DrydenWindModel:
    """
    Dryden风场湍流模型
    
    基于MIL-F-8785C规范实现，使用传递函数将白噪声成形为具有特定功率谱密度的湍流。
    
    参考：
    - MIL-F-8785C
    - MIL-HDBK-1797
    
    属性：
        intensity: 湍流强度 (0-1)，对应轻度到重度湍流
    """
    
    # 湍流强度映射 (intensity 0-1 -> sigma_w at 1000ft in m/s)
    # 轻度: ~1 m/s, 中度: ~3 m/s, 重度: ~6 m/s
    SIGMA_W_20FT_LIGHT = 0.5    # m/s
    SIGMA_W_20FT_SEVERE = 6.0   # m/s
    
    def __init__(self, intensity: float = 0.0, dt: float = 1.0, wingspan: float = 35.0):
        """
        初始化Dryden风场模型
        
        Args:
            intensity: 湍流强度 (0-1)
            dt: 时间步长（秒）
            wingspan: 飞机翼展（米），用于角速度湍流计算
        """
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.dt = dt
        self.wingspan = wingspan
        
        # 滤波器状态（每个轴的历史状态）
        self._u_state = np.zeros(2)  # 纵向速度滤波器状态
        self._v_state = np.zeros(3)  # 横向速度滤波器状态
        self._w_state = np.zeros(3)  # 垂向速度滤波器状态
        
        # 上一次的输出（用于平滑）
        self._last_output = np.zeros(3)
        
    def set_intensity(self, intensity: float):
        """设置湍流强度"""
        self.intensity = np.clip(intensity, 0.0, 1.0)
        
    def reset(self):
        """重置滤波器状态"""
        self._u_state = np.zeros(2)
        self._v_state = np.zeros(3)
        self._w_state = np.zeros(3)
        self._last_output = np.zeros(3)
    
    def _get_turbulence_params(self, altitude: float) -> Tuple[float, float, float, float, float, float]:
        """
        计算湍流参数（尺度长度和强度）
        
        基于MIL-F-8785C规范：
        - 低空 (h < 1000ft): L随高度变化
        - 中高空 (h >= 1000ft): L固定
        
        Args:
            altitude: 高度（米）
            
        Returns:
            (L_u, L_v, L_w, sigma_u, sigma_v, sigma_w)
        """
        h_ft = altitude * 3.28084  # 转换为英尺
        h_ft = max(h_ft, 10)  # 最小高度10ft，避免除零
        
        # 尺度长度计算 (MIL-F-8785C)
        if h_ft < 1000:
            # 低空模型
            L_w = h_ft  # ft
            L_u = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2
            L_v = L_u
        else:
            # 中高空模型
            L_u = 1750.0  # ft
            L_v = 1750.0  # ft
            L_w = 1750.0  # ft
        
        # 转换为米
        L_u *= 0.3048
        L_v *= 0.3048
        L_w *= 0.3048
        
        # 湍流强度计算
        # 基于20ft高度的风速，线性插值强度
        sigma_w_20 = self.SIGMA_W_20FT_LIGHT + self.intensity * (
            self.SIGMA_W_20FT_SEVERE - self.SIGMA_W_20FT_LIGHT)
        
        if h_ft < 1000:
            # 低空：sigma随高度变化
            sigma_w = sigma_w_20 * (h_ft / 20) ** 0.5
            sigma_u = sigma_w / (0.177 + 0.000823 * h_ft) ** 0.4
            sigma_v = sigma_u
        else:
            # 中高空：固定强度
            sigma_w = sigma_w_20 * 2.0  # 高空湍流放大
            sigma_u = sigma_w
            sigma_v = sigma_w
        
        return L_u, L_v, L_w, sigma_u, sigma_v, sigma_w
    
    def update(self, altitude: float, airspeed: float) -> Tuple[float, float, float]:
        """
        更新风场湍流，返回阵风速度分量
        
        Args:
            altitude: 当前高度（米）
            airspeed: 真空速（m/s）
            
        Returns:
            (u_gust, v_gust, w_gust): 纵向、横向、垂向阵风速度（m/s）
        """
        if self.intensity <= 0 or airspeed < 1:
            return 0.0, 0.0, 0.0
        
        # 获取湍流参数
        L_u, L_v, L_w, sigma_u, sigma_v, sigma_w = self._get_turbulence_params(altitude)
        
        V = max(airspeed, 1.0)
        
        # 生成白噪声输入
        noise_u = np.random.randn()
        noise_v = np.random.randn()
        noise_w = np.random.randn()
        
        # 离散时间常数
        tau_u = L_u / V
        tau_v = L_v / V
        tau_w = L_w / V
        
        # 一阶低通滤波器离散化 (纵向u)
        # H_u(s) = sigma_u * sqrt(2*L_u/(pi*V)) * 1/(1 + tau_u*s)
        alpha_u = self.dt / (tau_u + self.dt)
        K_u = sigma_u * np.sqrt(2 * L_u / (np.pi * V))
        
        u_gust = (1 - alpha_u) * self._u_state[0] + alpha_u * K_u * noise_u
        self._u_state[0] = u_gust
        
        # 二阶滤波器离散化 (横向v和垂向w)
        # 使用简化的一阶近似 + 导数项
        alpha_v = self.dt / (tau_v + self.dt)
        K_v = sigma_v * np.sqrt(L_v / (np.pi * V))
        
        # 使用二阶递归
        v_filtered = (1 - alpha_v) * self._v_state[0] + alpha_v * noise_v
        self._v_state[1] = self._v_state[0]
        self._v_state[0] = v_filtered
        
        # 添加导数项 (sqrt(3) * tau * s)
        v_derivative = (self._v_state[0] - self._v_state[1]) / self.dt * np.sqrt(3) * tau_v
        v_gust = K_v * (v_filtered + v_derivative * 0.3)
        
        # 垂向w (类似v)
        alpha_w = self.dt / (tau_w + self.dt)
        K_w = sigma_w * np.sqrt(L_w / (np.pi * V))
        
        w_filtered = (1 - alpha_w) * self._w_state[0] + alpha_w * noise_w
        self._w_state[1] = self._w_state[0]
        self._w_state[0] = w_filtered
        
        w_derivative = (self._w_state[0] - self._w_state[1]) / self.dt * np.sqrt(3) * tau_w
        w_gust = K_w * (w_filtered + w_derivative * 0.3)
        
        # 平滑输出（避免突变）
        smooth_factor = 0.7
        u_gust = smooth_factor * self._last_output[0] + (1 - smooth_factor) * u_gust
        v_gust = smooth_factor * self._last_output[1] + (1 - smooth_factor) * v_gust
        w_gust = smooth_factor * self._last_output[2] + (1 - smooth_factor) * w_gust
        
        self._last_output = np.array([u_gust, v_gust, w_gust])
        
        return u_gust, v_gust, w_gust


# ============================================================================
# 气动摄动模型
# ============================================================================

class AeroPerturbation:
    """
    气动摄动模型
    
    模拟由于湍流引起的额外气动力和力矩扰动
    """
    
    def __init__(self, intensity: float = 0.0, dt: float = 1.0):
        """
        初始化气动摄动模型
        
        Args:
            intensity: 摄动强度 (0-1)
            dt: 时间步长（秒）
        """
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.dt = dt
        self._last_perturbation = np.zeros(6)  # [Fx, Fy, Fz, Mx, My, Mz]
        
    def set_intensity(self, intensity: float):
        """设置摄动强度"""
        self.intensity = np.clip(intensity, 0.0, 1.0)
        
    def update(self, dynamic_pressure: float, wing_area: float, 
               wingspan: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算气动摄动
        
        Args:
            dynamic_pressure: 动压 (Pa)
            wing_area: 机翼面积 (m²)
            wingspan: 翼展 (m)
            
        Returns:
            (forces, moments): 力扰动[Fx,Fy,Fz](N), 力矩扰动[Mx,My,Mz](N·m)
        """
        if self.intensity <= 0:
            return np.zeros(3), np.zeros(3)
        
        # 扰动系数（基于强度）
        # 最大摄动为基础气动力的 5%
        force_scale = 0.05 * self.intensity * dynamic_pressure * wing_area
        moment_scale = 0.02 * self.intensity * dynamic_pressure * wing_area * wingspan
        
        # 生成相关随机扰动（低频）
        noise = np.random.randn(6)
        
        perturbation = np.array([
            force_scale * noise[0],   # Fx
            force_scale * noise[1],   # Fy
            force_scale * noise[2],   # Fz
            moment_scale * noise[3],  # Mx (滚转)
            moment_scale * noise[4],  # My (俯仰)
            moment_scale * noise[5],  # Mz (偏航)
        ])
        
        # 平滑处理
        smooth_factor = 0.8
        perturbation = smooth_factor * self._last_perturbation + (1 - smooth_factor) * perturbation
        self._last_perturbation = perturbation
        
        return perturbation[:3], perturbation[3:]


# ============================================================================
# IMU 量测噪声
# ============================================================================

class IMUNoise:
    """
    IMU量测噪声模型
    
    支持多种噪声类型：
    - white: 白噪声
    - flicker: 闪烁噪声（偶发的突变）
    - drift: 漂移噪声（累积性偏差）
    - colored: 有色噪声（时间相关）
    """
    
    # 大幅增大噪声范围以便效果明显
    ACC_NOISE_MIN = 0.1    # m/s²
    ACC_NOISE_MAX = 2.0    # m/s²
    GYRO_NOISE_MIN = 0.05  # deg/s
    GYRO_NOISE_MAX = 1.0   # deg/s
    
    def __init__(self, intensity: float = 0.0, noise_type: str = 'white',
                 drift_rate: float = 0.001, flicker_prob: float = 0.05,
                 flicker_scale: float = 3.0, colored_alpha: float = 0.9):
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.noise_type = noise_type
        self.drift_rate = drift_rate
        self.flicker_prob = flicker_prob
        self.flicker_scale = flicker_scale
        self.colored_alpha = colored_alpha
        
        # 状态变量
        self._acc_drift = np.zeros(3)
        self._gyro_drift = np.zeros(3)
        self._acc_colored = np.zeros(3)
        self._gyro_colored = np.zeros(3)
        self._step = 0
        
    def set_intensity(self, intensity: float):
        self.intensity = np.clip(intensity, 0.0, 1.0)
    
    def reset(self):
        self._acc_drift = np.zeros(3)
        self._gyro_drift = np.zeros(3)
        self._acc_colored = np.zeros(3)
        self._gyro_colored = np.zeros(3)
        self._step = 0
    
    @property
    def acc_sigma(self) -> float:
        return self.ACC_NOISE_MIN + self.intensity * (self.ACC_NOISE_MAX - self.ACC_NOISE_MIN)
    
    @property
    def gyro_sigma(self) -> float:
        return self.GYRO_NOISE_MIN + self.intensity * (self.GYRO_NOISE_MAX - self.GYRO_NOISE_MIN)
    
    def apply(self, true_acc: np.ndarray, true_gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """添加IMU噪声"""
        if self.intensity <= 0:
            return true_acc.copy(), true_gyro.copy()
        
        self._step += 1
        
        if self.noise_type == 'white':
            # 白噪声
            acc_noise = np.random.randn(3) * self.acc_sigma
            gyro_noise = np.random.randn(3) * self.gyro_sigma
            
        elif self.noise_type == 'flicker':
            # 闪烁噪声：大部分时间是小噪声，偶尔有大的突变
            if np.random.rand() < self.flicker_prob:
                scale = self.flicker_scale
            else:
                scale = 1.0
            acc_noise = np.random.randn(3) * self.acc_sigma * scale
            gyro_noise = np.random.randn(3) * self.gyro_sigma * scale
            
        elif self.noise_type == 'drift':
            # 漂移噪声：累积性的偏差
            self._acc_drift += np.random.randn(3) * self.drift_rate * self.acc_sigma
            self._gyro_drift += np.random.randn(3) * self.drift_rate * self.gyro_sigma
            acc_noise = self._acc_drift + np.random.randn(3) * self.acc_sigma * 0.3
            gyro_noise = self._gyro_drift + np.random.randn(3) * self.gyro_sigma * 0.3
            
        elif self.noise_type == 'colored':
            # 有色噪声：AR(1)模型
            self._acc_colored = self.colored_alpha * self._acc_colored + \
                               np.sqrt(1 - self.colored_alpha**2) * np.random.randn(3) * self.acc_sigma
            self._gyro_colored = self.colored_alpha * self._gyro_colored + \
                                np.sqrt(1 - self.colored_alpha**2) * np.random.randn(3) * self.gyro_sigma
            acc_noise = self._acc_colored
            gyro_noise = self._gyro_colored
        else:
            acc_noise = np.random.randn(3) * self.acc_sigma
            gyro_noise = np.random.randn(3) * self.gyro_sigma
        
        return true_acc + acc_noise, true_gyro + gyro_noise



# ============================================================================
# GPS/GNSS 量测噪声
# ============================================================================

class GPSNoise:
    """
    GPS/GNSS量测噪声模型
    
    支持多种噪声类型：
    - white: 白噪声
    - flicker: 闪烁噪声
    - drift: 漂移噪声
    - colored: 有色噪声
    """
    
    # 大幅增大噪声范围
    POS_NOISE_MIN = 10.0   # m
    POS_NOISE_MAX = 100.0  # m
    VEL_NOISE_MIN = 0.2    # m/s
    VEL_NOISE_MAX = 5.0    # m/s
    
    def __init__(self, intensity: float = 0.0, noise_type: str = 'white',
                 drift_rate: float = 0.001, flicker_prob: float = 0.05,
                 flicker_scale: float = 3.0, colored_alpha: float = 0.9):
        self.intensity = np.clip(intensity, 0.0, 1.0)
        self.noise_type = noise_type
        self.drift_rate = drift_rate
        self.flicker_prob = flicker_prob
        self.flicker_scale = flicker_scale
        self.colored_alpha = colored_alpha
        
        # 状态变量
        self._pos_drift = np.zeros(3)
        self._vel_drift = np.zeros(3)
        self._pos_colored = np.zeros(3)
        self._vel_colored = np.zeros(3)
        self._step = 0
        
    def set_intensity(self, intensity: float):
        self.intensity = np.clip(intensity, 0.0, 1.0)
    
    def reset(self):
        self._pos_drift = np.zeros(3)
        self._vel_drift = np.zeros(3)
        self._pos_colored = np.zeros(3)
        self._vel_colored = np.zeros(3)
        self._step = 0
    
    @property
    def pos_sigma(self) -> float:
        return self.POS_NOISE_MIN + self.intensity * (self.POS_NOISE_MAX - self.POS_NOISE_MIN)
    
    @property
    def vel_sigma(self) -> float:
        return self.VEL_NOISE_MIN + self.intensity * (self.VEL_NOISE_MAX - self.VEL_NOISE_MIN)
    
    def apply(self, true_lat: float, true_lon: float, true_alt: float,
              true_vel: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """添加GPS噪声"""
        if self.intensity <= 0:
            return true_lat, true_lon, true_alt, true_vel.copy()
        
        self._step += 1
        R_earth = 6371000
        
        if self.noise_type == 'white':
            pos_noise = np.random.randn(3) * self.pos_sigma
            vel_noise = np.random.randn(3) * self.vel_sigma
            
        elif self.noise_type == 'flicker':
            if np.random.rand() < self.flicker_prob:
                scale = self.flicker_scale
            else:
                scale = 1.0
            pos_noise = np.random.randn(3) * self.pos_sigma * scale
            vel_noise = np.random.randn(3) * self.vel_sigma * scale
            
        elif self.noise_type == 'drift':
            self._pos_drift += np.random.randn(3) * self.drift_rate * self.pos_sigma
            self._vel_drift += np.random.randn(3) * self.drift_rate * self.vel_sigma
            pos_noise = self._pos_drift + np.random.randn(3) * self.pos_sigma * 0.3
            vel_noise = self._vel_drift + np.random.randn(3) * self.vel_sigma * 0.3
            
        elif self.noise_type == 'colored':
            self._pos_colored = self.colored_alpha * self._pos_colored + \
                               np.sqrt(1 - self.colored_alpha**2) * np.random.randn(3) * self.pos_sigma
            self._vel_colored = self.colored_alpha * self._vel_colored + \
                               np.sqrt(1 - self.colored_alpha**2) * np.random.randn(3) * self.vel_sigma
            pos_noise = self._pos_colored
            vel_noise = self._vel_colored
        else:
            pos_noise = np.random.randn(3) * self.pos_sigma
            vel_noise = np.random.randn(3) * self.vel_sigma
        
        # 转换为经纬度增量
        dlat = np.degrees(pos_noise[0] / R_earth)
        dlon = np.degrees(pos_noise[1] / (R_earth * np.cos(np.radians(true_lat))))
        
        noisy_lat = true_lat + dlat
        noisy_lon = true_lon + dlon
        noisy_alt = max(0, true_alt + pos_noise[2] * 1.5)
        noisy_vel = true_vel + vel_noise
        
        return noisy_lat, noisy_lon, noisy_alt, noisy_vel



# ============================================================================
# 噪声管理器
# ============================================================================

class NoiseManager:
    """
    噪声管理器
    
    统一管理所有噪声模型
    """
    
    def __init__(self, config: Optional[NoiseConfig] = None, dt: float = 1.0, 
                 wingspan: float = 35.0):
        """
        初始化噪声管理器
        
        Args:
            config: 噪声配置
            dt: 时间步长（秒）
            wingspan: 飞机翼展（米）
        """
        self.config = config or NoiseConfig()
        self.dt = dt
        self.wingspan = wingspan
        
        # 初始化各噪声模型
        self.wind_model = DrydenWindModel(
            intensity=self.config.wind_intensity,
            dt=dt,
            wingspan=wingspan
        )
        self.aero_pert = AeroPerturbation(
            intensity=self.config.aero_perturbation,
            dt=dt
        )
        self.imu_noise = IMUNoise(
            intensity=self.config.imu_noise,
            noise_type=self.config.imu_noise_type,
            drift_rate=self.config.imu_drift_rate,
            flicker_prob=self.config.imu_flicker_prob,
            flicker_scale=self.config.imu_flicker_scale,
            colored_alpha=self.config.imu_colored_alpha
        )
        self.gps_noise = GPSNoise(
            intensity=self.config.gps_noise,
            noise_type=self.config.gps_noise_type,
            drift_rate=self.config.gps_drift_rate,
            flicker_prob=self.config.gps_flicker_prob,
            flicker_scale=self.config.gps_flicker_scale,
            colored_alpha=self.config.gps_colored_alpha
        )
        
    def update_config(self, config: NoiseConfig):
        """更新噪声配置"""
        self.config = config
        self.wind_model.set_intensity(config.wind_intensity)
        self.aero_pert.set_intensity(config.aero_perturbation)
        # 重新初始化IMU和GPS噪声模型，使用各自独立的配置
        self.imu_noise = IMUNoise(
            intensity=config.imu_noise,
            noise_type=config.imu_noise_type,
            drift_rate=config.imu_drift_rate,
            flicker_prob=config.imu_flicker_prob,
            flicker_scale=config.imu_flicker_scale,
            colored_alpha=config.imu_colored_alpha
        )
        self.gps_noise = GPSNoise(
            intensity=config.gps_noise,
            noise_type=config.gps_noise_type,
            drift_rate=config.gps_drift_rate,
            flicker_prob=config.gps_flicker_prob,
            flicker_scale=config.gps_flicker_scale,
            colored_alpha=config.gps_colored_alpha
        )
        
    def get_wind_gust(self, altitude: float, airspeed: float) -> Tuple[float, float, float]:
        """获取阵风速度"""
        return self.wind_model.update(altitude, airspeed)
    
    def get_aero_perturbation(self, dynamic_pressure: float, wing_area: float,
                               wingspan: float) -> Tuple[np.ndarray, np.ndarray]:
        """获取气动摄动"""
        return self.aero_pert.update(dynamic_pressure, wing_area, wingspan)
    
    def apply_imu_noise(self, true_acc: np.ndarray, 
                        true_gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用IMU噪声"""
        return self.imu_noise.apply(true_acc, true_gyro)

    def apply_attitude_noise(self, true_pitch: float, true_roll: float, true_heading: float) -> Tuple[float, float, float]:
        """
        应用姿态角噪声 (简化模拟AHRS误差)
        使用IMU的陀螺仪噪声参数来生成角度噪声
        """
        # 将角度打包成向量
        true_att = np.array([true_pitch, true_roll, true_heading])
        # 构造虚拟加速度 (不使用)
        dummy_acc = np.zeros(3)
        
        # 复用IMU噪声模型
        # 注意：这会推进imu_noise的内部状态 (如漂移、有色噪声等)
        # 我们忽略加速度噪声，仅使用陀螺仪部分作为角度噪声
        _, noisy_att_vec = self.imu_noise.apply(dummy_acc, true_att)
        
        return noisy_att_vec[0], noisy_att_vec[1], noisy_att_vec[2]
    
    def apply_gps_noise(self, true_lat: float, true_lon: float, true_alt: float,
                        true_vel: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """应用GPS噪声"""
        return self.gps_noise.apply(true_lat, true_lon, true_alt, true_vel)
    
    def reset(self):
        """重置所有噪声模型状态"""
        self.wind_model.reset()
        self.aero_pert._last_perturbation = np.zeros(6)


# ============================================================================
# 旧版兼容 - 保留原有的噪声类
# ============================================================================

class BaseNoiseModel():
    """
    基础噪声模型（保留向后兼容）
    """
    def __init__(self, frame: int) -> None:
        self.name = 'base_noise'
        self.frame = frame
        self.noise = np.zeros((frame, 6))

    def __call__(self) -> np.ndarray:
        return self.noise


class GaussianNoise(BaseNoiseModel):
    """高斯白噪声"""
    def __init__(self, frame: int, mean: np.ndarray, sigma: np.ndarray) -> None:
        super().__init__(frame)
        self.name = 'GaussianNoise'
        self.noise = np.zeros((frame, 6))
        for i in range(frame): 
            self.noise[i,:] = np.random.normal(mean, sigma, (1, 6))


class FlickerNoise(BaseNoiseModel):
    """闪变噪声"""
    def __init__(self, frame: int, mean: np.ndarray, sigma_1: np.ndarray, 
                 sigma_2: np.ndarray, flicker_prob: float) -> None:
        super().__init__(frame)
        self.name = 'FlickerNoise'
        self.noise = np.zeros((frame, 6))
        for i in range(frame): 
            flag = np.random.uniform(0, 1)
            if flag > flicker_prob:
                self.noise[i,:] = np.random.normal(mean, sigma_1, (1, 6))
            else:
                self.noise[i,:] = np.random.normal(mean, sigma_2, (1, 6))


class TimeVaryNoise(BaseNoiseModel):
    """时变噪声"""
    def __init__(self, frame: int, mean: np.ndarray, sigma: np.ndarray, T: int) -> None:
        super().__init__(frame)
        self.name = 'TimeVaryNoise'
        self.noise = np.zeros((frame, 6))
        for i in range(frame): 
            self.noise[i,:] = np.random.normal(mean, sigma*(1-np.sin(i/T)), (1, 6))


class DriftNoise(BaseNoiseModel):
    """漂移噪声"""
    def __init__(self, frame: int, mean: np.ndarray, sigma: np.ndarray, 
                 base: np.ndarray, dx: float) -> None:
        super().__init__(frame)
        self.name = 'DriftNoise'
        self.noise = np.zeros((frame, 6))
        for i in range(frame): 
            scale = dx * i * base
            self.noise[i,:] = np.random.normal(mean + scale, sigma, (1, 6))
