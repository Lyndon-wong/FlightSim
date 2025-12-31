"""
优化的六自由度动力学模型
实现基于真实飞机参数的高保真飞行器动力学仿真
"""
import numpy as np
from typing import Dict, Tuple
from .aerodynamics import AircraftParams, get_aircraft


class SixDOFModel:
    """
    六自由度飞行器动力学模型
    
    实现基于物理的飞行器运动仿真，包括：
    - 详细气动力计算（考虑马赫数、雷诺数效应）
    - 发动机推力模型（考虑高度、速度影响）
    - 姿态动力学
    - 位置更新
    - 燃油消耗
    """
    
    def __init__(self, aircraft_type: str, start_lat: float, start_lon: float, 
                 start_alt: float, start_heading: float, dt: float = 1.0):
        """
        初始化六自由度模型
        
        Args:
            aircraft_type: 飞机型号（从plans.csv中加载）
            start_lat: 起始纬度（度）
            start_lon: 起始经度（度）
            start_alt: 起始高度（米）
            start_heading: 起始航向（度）
            dt: 时间步长（秒）
        """
        self.params: AircraftParams = get_aircraft(aircraft_type)
        self.dt = dt
        
        # 位置状态
        self.lat = start_lat
        self.lon = start_lon
        self.alt = start_alt
        
        # 速度状态（m/s）
        self.tas = 0.0  # 真空速 (True Airspeed)
        self.ias = 0.0  # 指示空速 (Indicated Airspeed)
        self.gs = 0.0   # 地速 (Ground Speed)
        self.v_vertical = 0.0  # 垂直速度
        self.mach = 0.0  # 马赫数
        
        # 姿态状态（度）
        self.heading = start_heading  # 航向
        self.pitch = 0.0  # 俯仰角
        self.roll = 0.0  # 滚转角
        self.gamma = 0.0  # 航迹角（爬升角）
        self.alpha = 0.0  # 攻角
        self.beta = 0.0  # 侧滑角
        
        # 角速度（deg/s）
        self.p = 0.0  # 滚转角速度
        self.q = 0.0  # 俯仰角速度
        self.r = 0.0  # 偏航角速度
        
        # 配置状态
        self.flaps_idx = 0  # 襟翼位置索引 (0=收起, 1=起飞, 2=着陆)
        self.gear_down = True  # 起落架状态
        self.spoilers = False  # 扰流板状态
        
        # 质量状态
        self.current_mass_kg = self.params.typical_mass_kg
        self.fuel_kg = self.params.fuel_capacity_kg * 0.7  # 初始燃油70%
        
        # 力和力矩
        self.lift = 0.0  # 升力 (N)
        self.drag = 0.0  # 阻力 (N)
        self.thrust = 0.0  # 推力 (N)
        
        # 性能参数
        self.load_factor = 1.0  # 载荷因子 (g)
        
    def set_config(self, flaps_idx: int, gear_down: bool, spoilers: bool = False):
        """
        设置飞机配置
        
        Args:
            flaps_idx: 襟翼位置索引（0=收起，1=起飞，2=着陆）
            gear_down: 起落架是否放下
            spoilers: 扰流板是否展开
        """
        self.flaps_idx = np.clip(flaps_idx, 0, 2)
        self.gear_down = gear_down
        self.spoilers = spoilers
    
    def _get_atmosphere(self, altitude: float) -> Tuple[float, float, float]:
        """
        计算大气参数（国际标准大气ISA）
        
        Args:
            altitude: 高度（米）
            
        Returns:
            (密度 kg/m³, 温度 K, 音速 m/s)
        """
        # 海平面标准值
        rho_0 = 1.225  # kg/m³
        T_0 = 288.15   # K
        a_0 = 340.294  # m/s
        
        # 对流层（0-11km）
        if altitude <= 11000:
            T = T_0 - 0.0065 * altitude
            rho = rho_0 * (T / T_0) ** 4.256
        # 平流层下层（11-20km）
        elif altitude <= 20000:
            T = 216.65
            rho = 0.3639 * np.exp(-(altitude - 11000) / 6341.6)
        # 平流层（20-32km）
        else:
            T = 216.65 + 0.001 * (altitude - 20000)
            rho = 0.0880 * (T / 216.65) ** (-35.163)
        
        # 音速
        a = np.sqrt(1.4 * 287.05 * T)
        
        return max(rho, 0.01), T, a
    
    def _calculate_reynolds_correction(self, altitude: float) -> float:
        """
        雷诺数修正系数（简化模型）
        
        Args:
            altitude: 高度（米）
            
        Returns:
            修正系数（0.9-1.0）
        """
        # 高空雷诺数降低，导致阻力略微增加
        return 1.0 - 0.05 * min(altitude / 12000, 1.0)
    
    def _calculate_mach_correction(self, mach: float) -> Tuple[float, float]:
        """
        马赫数修正（压缩性效应）
        
        Args:
            mach: 马赫数
            
        Returns:
            (升力修正系数, 阻力修正系数)
        """
        # Prandtl-Glauert修正
        if mach < 0.3:
            cl_corr = 1.0
            cd_corr = 1.0
        elif mach < 0.7:
            # 亚音速修正
            beta = np.sqrt(1 - mach**2)
            cl_corr = 1.0 / beta
            cd_corr = 1.0
        elif mach < 0.9:
            # 跨音速阻力上升
            cl_corr = 1.0 / np.sqrt(1 - 0.7**2)
            cd_corr = 1.0 + 10 * (mach - 0.7)**2
        else:
            # 超音速（不常见于民航）
            cl_corr = 1.0
            cd_corr = 2.0
        
        return cl_corr, cd_corr
    
    def _calculate_aero(self, alpha: float, flaps: int, gear: bool, 
                       spoilers: bool, mach: float) -> Tuple[float, float]:
        """
        计算升力和阻力系数（考虑各种修正）
        
        Args:
            alpha: 攻角（度）
            flaps: 襟翼位置索引
            gear: 起落架是否放下
            spoilers: 扰流板是否展开
            mach: 马赫数
            
        Returns:
            (升力系数, 阻力系数)
        """
        aero = self.params.aero
        stall_ang = aero.alpha_stall[flaps]
        
        # 基本升力系数
        if alpha < stall_ang:
            cl_basic = aero.cl_alpha * alpha + [0.0, 0.4, 0.8][flaps]
        else:
            # 失速后升力下降
            cl_basic = aero.cl_max[flaps] * max(0.3, 1.0 - (alpha - stall_ang) * 0.08)
        
        # 马赫数修正
        cl_mach_corr, cd_mach_corr = self._calculate_mach_correction(mach)
        cl = cl_basic * cl_mach_corr
        
        # 基本阻力系数（零升阻力 + 诱导阻力）
        cd_basic = aero.cd_0[flaps] + aero.k * (cl**2)
        
        # 附加阻力
        if gear:
            cd_basic += aero.gear_drag
        if spoilers:
            cd_basic += 0.15  # 扰流板大幅增加阻力
            cl *= 0.7  # 扰流板减少升力
        
        # 马赫数修正
        cd = cd_basic * cd_mach_corr
        
        return cl, cd
    
    def _calculate_thrust(self, throttle_pct: float, altitude: float, 
                         mach: float, temperature: float) -> float:
        """
        计算发动机推力（考虑高度和速度影响）
        
        Args:
            throttle_pct: 油门百分比（0-1）
            altitude: 高度（米）
            mach: 马赫数
            temperature: 温度（K）
            
        Returns:
            推力（N）
        """
        # 基准推力
        thrust_max = self.params.max_thrust_n
        
        # 高度修正（密度比）
        rho, _, _ = self._get_atmosphere(altitude)
        sigma = rho / 1.225  # 密度比
        
        # 涡扇发动机推力随高度和速度的变化
        # 简化模型：T = T_0 * sigma^0.7 * (1 - 0.15*M)
        altitude_factor = sigma ** 0.7
        mach_factor = max(0.5, 1.0 - 0.15 * mach)
        
        # 温度修正（简化）
        temp_factor = np.sqrt(288.15 / temperature)
        
        thrust = thrust_max * throttle_pct * altitude_factor * mach_factor * temp_factor
        
        return max(0, thrust)
    
    def _calculate_fuel_flow(self, thrust: float, altitude: float) -> float:
        """
        计算燃油流量（改进模型 - 修正TSFC单位）
        
        Args:
            thrust: 当前总推力（N）- 所有发动机
            altitude: 高度（米）
            
        Returns:
            燃油流量（kg/s）- 所有发动机总和
            
        注：CFM56-5B (A320使用) 巡航TSFC约0.50-0.55 lb/(lbf·h)
            转换为SI单位：1 lb/(lbf·h) ≈ 0.0283 kg/(N·h)
            因此SI单位的TSFC约0.014-0.016 kg/(N·h)
        """
        # TSFC随高度和飞行条件变化（单位: kg/(N·h)）
        # 考虑真实飞行数据：A320巡航时~2500kg/h，推力~65-75kN
        # 因此TSFC ≈ 2500/70000*3600 ≈ 0.0357 kg/(N·h)
        if altitude < 3000:  # 低空（起飞/爬升）
            tsfc_base = 0.030  # 起飞TSFC较高
        elif altitude < 8000:  # 中空
            tsfc_base = 0.030 + 0.005 * (altitude - 3000) / 5000
        else:  # 高空巡航（>8km）
            tsfc_base = 0.035 + 0.003 * min((altitude - 8000) / 4000, 1.0)
            # 巡航时约0.035-0.038 kg/(N·h)
        
        # 转换为 kg/(N·s)
        tsfc = tsfc_base / 3600
        
        # 总燃油流量 = 总推力 * TSFC
        fuel_flow = thrust * tsfc
        
        return fuel_flow
    
    def update(self, throttle_pct: float, target_pitch: float, target_roll: float):
        """
        更新动力学状态
        
        Args:
            throttle_pct: 油门百分比（0-1）
            target_pitch: 目标俯仰角（度）
            target_roll: 目标滚转角（度）
        """
        # 获取大气参数
        rho, temp, sound_speed = self._get_atmosphere(self.alt)
        
        # 计算马赫数和指示空速
        self.mach = self.tas / sound_speed
        # 指示空速修正（简化）
        self.ias = self.tas * np.sqrt(rho / 1.225)
        
        # 姿态更新（一阶惯性环节）
        max_pitch_rate = self.params.max_pitch_rate * self.dt
        max_roll_rate = self.params.max_roll_rate * self.dt
        
        dp = np.clip(target_pitch - self.pitch, -max_pitch_rate, max_pitch_rate)
        dr = np.clip(target_roll - self.roll, -max_roll_rate, max_roll_rate)
        
        self.pitch += dp
        self.roll += dr
        
        # 攻角计算
        self.alpha = self.pitch - np.degrees(self.gamma)
        
        # 气动力计算
        cl, cd = self._calculate_aero(
            self.alpha, self.flaps_idx, self.gear_down, 
            self.spoilers, self.mach
        )
        
        # 动压
        q_bar = 0.5 * rho * max(self.tas, 1.0)**2
        
        # 气动力
        self.lift = cl * q_bar * self.params.wing_area_m2
        self.drag = cd * q_bar * self.params.wing_area_m2
        
        # 推力
        self.thrust = self._calculate_thrust(throttle_pct, self.alt, self.mach, temp)
        
        # 燃油消耗
        fuel_flow = self._calculate_fuel_flow(self.thrust, self.alt)
        self.fuel_kg = max(0, self.fuel_kg - fuel_flow * self.dt)
        
        # 更新质量
        self.current_mass_kg = self.params.oew_kg + self.fuel_kg + \
                              0.5 * self.params.max_payload_kg
        
        # 力平衡方程
        weight = self.current_mass_kg * 9.81
        
        # 纵向运动方程
        # 沿航迹方向: T - D - W*sin(gamma) = m*dV/dt
        acc_tangent = (self.thrust - self.drag - weight * np.sin(self.gamma)) / self.current_mass_kg
        self.tas = max(0, self.tas + acc_tangent * self.dt)
        
        # 垂直于航迹方向: L - W*cos(gamma) = m*V*dgamma/dt
        if self.tas > 20:
            lift_vert = self.lift * np.cos(np.radians(self.roll))
            d_gamma = (lift_vert - weight * np.cos(self.gamma)) / (self.current_mass_kg * self.tas)
            self.gamma += d_gamma * self.dt
            # self.gamma = np.clip(self.gamma, -0.15, 0.15)  # 移除硬限制，避免动力学不连续
            
            # 载荷因子
            self.load_factor = lift_vert / weight
            
            # 转弯（侧向运动）
            lift_horiz = self.lift * np.sin(np.radians(self.roll))
            turn_rate = lift_horiz / (self.current_mass_kg * self.tas)
            self.heading = (self.heading + np.degrees(turn_rate) * self.dt) % 360
        
        # 位置更新
        self.v_vertical = self.tas * np.sin(self.gamma)
        v_horiz = self.tas * np.cos(self.gamma)
        
        # 北向和东向速度分量
        vn = v_horiz * np.cos(np.radians(self.heading))
        ve = v_horiz * np.sin(np.radians(self.heading))
        
        # 地速
        self.gs = np.sqrt(vn**2 + ve**2)
        
        # 经纬度更新（球面地球模型）
        R_earth = 6371000  # 地球半径（米）
        self.lat += np.degrees(vn * self.dt / R_earth)
        self.lon += np.degrees(ve * self.dt / (R_earth * np.cos(np.radians(self.lat))))
        self.alt = max(0, self.alt + self.v_vertical * self.dt)
        
        # 地面运动处理
        if self.alt <= 0:
            self.alt = 0
            self.v_vertical = 0
            self.gamma = 0
            
            # 地面摩擦和滚动阻力
            if self.tas > 0:
                # 简化地面阻力模型
                ground_friction = 0.02  # 滚动摩擦系数
                ground_drag = ground_friction * weight
                
                # 地面加速度
                if self.thrust > ground_drag:
                    # 加速
                    ground_acc = (self.thrust - ground_drag) / self.current_mass_kg
                    self.tas = self.tas + ground_acc * self.dt
                else:
                    # 减速
                    self.tas = max(0, self.tas - 2 * self.dt)
    
    def get_state(self) -> Dict:
        """
        获取当前状态
        
        Returns:
            状态字典，包含位置、速度、姿态等详细信息
        """
        return {
            # 位置
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            
            # 速度
            "tas": self.tas,  # 真空速 (m/s)
            "ias": self.ias,  # 指示空速 (m/s)
            "gs": self.gs,    # 地速 (m/s)
            "speed_v": self.v_vertical,  # 垂直速度 (m/s)
            "mach": self.mach,
            
            # 姿态
            "heading": self.heading,
            "pitch": self.pitch,
            "roll": self.roll,
            "gamma": self.gamma,  # 航迹角
            "alpha": self.alpha,  # 攻角
            
            # 力
            "lift": self.lift,
            "drag": self.drag,
            "thrust": self.thrust,
            
            # 配置
            "flaps": self.flaps_idx,
            "gear": self.gear_down,
            "spoilers": self.spoilers,
            
            # 质量和性能
            "mass": self.current_mass_kg,
            "fuel": self.fuel_kg,
            "load_factor": self.load_factor,
        }
