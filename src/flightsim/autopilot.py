"""
改进的自动驾驶系统
实现标准民航飞行程序，包括：
- SID (Standard Instrument Departure) 标准离场程序
- 巡航导航
- STAR (Standard Terminal Arrival Route) 标准到达程序
- 五边进近程序（矩形航线）
- 详细的飞行阶段管理
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from enum import Enum
from .navigation import NavUtils
from .sixdof import SixDOFModel


class FlightPhase(Enum):
    """
    飞行阶段枚举（使用专业民航术语）
    """
    # 地面阶段
    TAXI = "TAXI"  # 滑行
    
    # 起飞阶段
    TAKEOFF_ROLL = "TAKEOFF_ROLL"  # 起飞滑跑
    ROTATION = "ROTATION"  # 抬轮
    INITIAL_CLIMB = "INITIAL_CLIMB"  # 初始爬升
    
    # 离场阶段
    DEPARTURE = "DEPARTURE"  # 离场（SID）
    CLIMB = "CLIMB"  # 爬升
    
    # 巡航阶段
    CRUISE = "CRUISE"  # 巡航
    
    # 下降阶段
    DESCENT = "DESCENT"  # 下降
    APPROACH_DESCENT = "APPROACH_DESCENT"  # 进近下降
    
    # 进近阶段（五边）
    DOWNWIND = "DOWNWIND"  # 三边（下风边）
    BASE = "BASE"  # 四边（基线边）
    FINAL = "FINAL"  # 五边（最后进近）
    
    # 着陆阶段
    FLARE = "FLARE"  # 拉平
    TOUCHDOWN = "TOUCHDOWN"  # 接地
    ROLLOUT = "ROLLOUT"  # 着陆滑跑
    
    # 复飞
    GO_AROUND = "GO_AROUND"  # 复飞


class BaseAutopilot(ABC):
    """自动驾驶基类"""
    
    def __init__(self, model: SixDOFModel):
        self.model = model
        self.phase = FlightPhase.TAXI
    
    @abstractmethod
    def update(self) -> Tuple[float, float, float]:
        """
        更新自动驾驶控制指令
        
        Returns:
            (throttle_pct, target_pitch, target_roll) 元组
        """
        pass
    
    def get_phase(self) -> FlightPhase:
        """获取当前飞行阶段"""
        return self.phase


class StandardAutopilot(BaseAutopilot):
    """
    标准自动驾驶系统
    实现完整的民航飞行程序，包括五边进近
    """
    
    def __init__(self, model: SixDOFModel, cruise_speed_mach: Optional[float] = None):
        super().__init__(model)
        
        # 航路点
        self.waypoints: List[Tuple[float, float]] = []
        self.current_wp_idx = 0
        
        # 目标参数
        self.target_alt = 0  # 目标高度（米）
        self.target_speed = 0  # 目标速度（m/s）
        self.target_heading = 0  # 目标航向（度）
        
        # 巡航参数（从飞机性能数据获取，可配置）
        perf = model.params.performance
        self.cruise_alt_m = perf.cruise_alt_ft * 0.3048  # 转换为米
        # 使用指定的巡航速度或默认最优速度
        actual_cruise_mach = cruise_speed_mach if cruise_speed_mach else perf.cruise_speed_mach
        self.cruise_speed_ms = actual_cruise_mach * 340 * 0.85  # 近似巡航速度
        self.cruise_speed_mach = actual_cruise_mach
        
        # 进近参数
        self.approach_speed_ms = perf.approach_speed_kts * 0.5144  # 节转m/s
        self.pattern_alt_m = 450  # 五边航线高度（米，约1500英尺）
        self.final_approach_alt_m = 150  # 五边起始高度（米，约500英尺）
        
        # 五边航线关键点
        self.pattern_entry_point: Optional[Tuple[float, float]] = None
        self.downwind_point: Optional[Tuple[float, float]] = None
        self.base_turn_point: Optional[Tuple[float, float]] = None
        self.final_turn_point: Optional[Tuple[float, float]] = None
        
        # 机场信息
        self.departure_airport: Optional[Tuple[float, float]] = None
        self.destination_airport: Optional[Tuple[float, float]] = None
        self.runway_heading = 0  # 跑道方向
        self.total_route_distance = 0  # 总航线距离（米）
        
        # 飞行阶段
        self.phase = FlightPhase.TAXI
        
        # 控制参数（基于航线占比）- 子类可覆盖
        self._init_tuning_params()
    
    def _init_tuning_params(self):
        """
        初始化控制调参参数 - 子类可覆盖以实现差异化控制
        """
        # 飞行阶段占比参数
        self.departure_distance_ratio = 0.05  # 离场阶段：前5%
        self.climb_distance_ratio = 0.15      # 爬升阶段：前15%
        self.descent_distance_ratio = 0.80    # 下降开始：80%位置
        self.approach_distance_ratio = 0.95   # 进近开始：95%位置
        
        # PID控制增益
        self.climb_rate_gain = 0.15   # 爬升率增益
        self.descent_rate_gain = 0.30 # 下降率增益
        self.max_climb_vs = 12.0      # 最大爬升率 m/s
        self.max_descent_vs = 15.0    # 最大下降率 m/s
        self.roll_gain = 1.5          # 滚转增益
        self.max_roll = 30.0          # 最大滚转角
        
        # 进近阶段专用参数
        self.glide_slope_angle = 3.0      # 下滑道角度（度）
        self.final_entry_dist_m = 12000   # 进入FINAL的距离阈值（米）
        self.final_entry_alt_margin = 80  # 高度容差（米）
        self.speed_decel_rate = 1.2       # 减速率（m/s per 2sec）
        self.approach_speed_factor = 1.25 # 进近速度系数（相对于着陆速度）
        self.final_speed_factor = 1.05    # 五边速度系数
        
    def load_route(self, waypoints: List[Tuple[float, float]], 
                   departure_alt: float = 10.0,
                   runway_heading: float = 0.0):
        """
        加载航路点
        
        Args:
            waypoints: 航路点列表，每个点为(lat, lon)元组
            departure_alt: 起飞机场高程（米）
            runway_heading: 跑道方向（度）
        """
        self.waypoints = waypoints
        self.current_wp_idx = 0
        
        if len(waypoints) >= 2:
            self.departure_airport = waypoints[0]
            self.destination_airport = waypoints[-1]
            self.runway_heading = runway_heading
            
            # 计算总航程距离（直线距离）
            self.total_route_distance = NavUtils.haversine_distance(
                waypoints[0][0], waypoints[0][1],
                waypoints[-1][0], waypoints[-1][1]
            )
            
            # 根据航程调整巡航高度
            perf = self.model.params.performance
            original_cruise_alt_m = perf.cruise_alt_ft * 0.3048
            
            # 短程航线特殊处理
            if self.total_route_distance < 300000:  # < 300km
                optimal_cruise_alt = min(original_cruise_alt_m, 3000 + self.total_route_distance * 0.01)
                self.cruise_alt_m = max(3000, optimal_cruise_alt)
            else:
                self.cruise_alt_m = original_cruise_alt_m
            
            # 根据航程动态调整阶段占比
            if self.total_route_distance < 500000:  # 短程 < 500km
                self.climb_distance_ratio = 0.20    # 爬升到20%
                self.descent_distance_ratio = 0.75  # 75%开始下降
                self.approach_distance_ratio = 0.90 # 90%进近
            elif self.total_route_distance < 2000000:  # 中程 < 2000km
                self.climb_distance_ratio = 0.15
                self.descent_distance_ratio = 0.80
                self.approach_distance_ratio = 0.95
            else:  # 长程 >= 2000km
                self.climb_distance_ratio = 0.10
                self.descent_distance_ratio = 0.85
                self.approach_distance_ratio = 0.94  # 从0.97改为0.94，增加进近距离
            
            # 计算五边航线
            self._setup_traffic_pattern()
    
    def _setup_traffic_pattern(self):
        """
        设置五边进近航线（标准左航线）
        
        五边航线结构：
        1. 一边（Upwind）：起飞后直线爬升
        2. 二边（Crosswind）：左转90度横风边
        3. 三边（Downwind）：平行跑道反向飞行
        4. 四边（Base）：左转90度基线边
        5. 五边（Final）：对准跑道最后进近
        
        尺寸设计（民航标准）：
        - 三边长度：8-10km（平行跑道反向飞行）
        - 航线宽度：3-5km（距离跑道中心线）
        - 五边长度：约5km（最后进近段）
        """
        if self.destination_airport is None:
            return
        
        dest_lat, dest_lon = self.destination_airport
        
        # 五边航线尺寸（扩大以适应民航飞机）
        pattern_length_m = 8000   # 三边长度8km
        pattern_width_m = 4000    # 航线宽度4km  
        final_approach_length_m = 5000  # 五边长度5km
        
        # 计算各关键点（相对于跑道方向）
        runway_hdg_rad = np.radians(self.runway_heading)
        
        # 三边（Downwind）- 平行跑道，反向，左侧
        # 位于跑道延长线后方8km，左侧4km处
        downwind_offset_lat = (pattern_width_m / 111132) * np.cos(runway_hdg_rad + np.pi/2)
        downwind_offset_lon = (pattern_width_m / (111132 * np.cos(np.radians(dest_lat)))) * \
                              np.sin(runway_hdg_rad + np.pi/2)
        
        # 三边中点（跑道后方8km）
        downwind_lat = dest_lat + downwind_offset_lat + \
                      (pattern_length_m / 111132) * np.cos(runway_hdg_rad + np.pi)
        downwind_lon = dest_lon + downwind_offset_lon + \
                      (pattern_length_m / (111132 * np.cos(np.radians(dest_lat)))) * \
                      np.sin(runway_hdg_rad + np.pi)
        
        self.downwind_point = (downwind_lat, downwind_lon)
        
        # 四边转弯点（Base turn）- 跑道后方，左侧4km
        base_lat = dest_lat + downwind_offset_lat
        base_lon = dest_lon + downwind_offset_lon
        self.base_turn_point = (base_lat, base_lon)
        
        # 五边转弯点（Final turn）- 跑道后方5km，中心线上
        final_lat = dest_lat - (final_approach_length_m / 111132) * np.cos(runway_hdg_rad)
        final_lon = dest_lon - (final_approach_length_m / (111132 * np.cos(np.radians(dest_lat)))) * \
                   np.sin(runway_hdg_rad)
        self.final_turn_point = (final_lat, final_lon)

    def update(self) -> Tuple[float, float, float]:
        """
        更新自动驾驶控制指令
        
        Returns:
            (throttle_pct, target_pitch, target_roll) 元组
        """
        state = self.model.get_state()
        
        # 更新飞行阶段
        self._update_phase(state)
        
        # 根据飞行阶段执行不同的控制逻辑
        if self.phase == FlightPhase.TAXI:
            return self._control_taxi()
        elif self.phase == FlightPhase.TAKEOFF_ROLL:
            return self._control_takeoff_roll()
        elif self.phase == FlightPhase.ROTATION:
            return self._control_rotation()
        elif self.phase == FlightPhase.INITIAL_CLIMB:
            return self._control_initial_climb()
        elif self.phase in [FlightPhase.DEPARTURE, FlightPhase.CLIMB]:
            return self._control_climb()
        elif self.phase == FlightPhase.CRUISE:
            return self._control_cruise()
        elif self.phase == FlightPhase.DESCENT:
            return self._control_descent()
        elif self.phase == FlightPhase.APPROACH_DESCENT:
            return self._control_approach_descent()
        elif self.phase == FlightPhase.DOWNWIND:
            return self._control_downwind()
        elif self.phase == FlightPhase.BASE:
            return self._control_base()
        elif self.phase == FlightPhase.FINAL:
            return self._control_final()
        elif self.phase == FlightPhase.FLARE:
            return self._control_flare()
        elif self.phase in [FlightPhase.TOUCHDOWN, FlightPhase.ROLLOUT]:
            return self._control_landing()
        else:
            return 0.0, 0.0, 0.0
    
    def _update_phase(self, state: dict):
        """更新飞行阶段状态机（基于航线占比）"""
        alt = state['alt']
        tas = state['tas']
        
        if self.destination_airport is None or self.total_route_distance == 0:
            return
        
        # 计算到目的地的2D距离（不考虑高度）
        dist_to_dest_2d = NavUtils.haversine_distance(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        # 计算到目的地的3D距离（考虑高度）
        horizontal_dist = dist_to_dest_2d
        vertical_dist = abs(alt - 10.0)  # 假设目标机场高度10m
        dist_to_dest_3d = np.sqrt(horizontal_dist**2 + vertical_dist**2)
        
        # 计算航线进度（已飞行的占比）
        dist_flown = self.total_route_distance - dist_to_dest_2d
        progress_ratio = dist_flown / self.total_route_distance if self.total_route_distance > 0 else 0
        progress_ratio = np.clip(progress_ratio, 0.0, 1.0)
        
        # 阶段转换逻辑
        if self.phase == FlightPhase.TAXI:
            if tas > 5:
                self.phase = FlightPhase.TAKEOFF_ROLL
                self.model.set_config(flaps_idx=1, gear_down=True)
                
        elif self.phase == FlightPhase.TAKEOFF_ROLL:
            v_r = self.approach_speed_ms * 1.1
            if tas >= v_r:
                self.phase = FlightPhase.ROTATION
                
        elif self.phase == FlightPhase.ROTATION:
            if alt > 15:
                self.phase = FlightPhase.INITIAL_CLIMB
                
        elif self.phase == FlightPhase.INITIAL_CLIMB:
            if alt > 500:
                self.phase = FlightPhase.DEPARTURE
                self.model.set_config(flaps_idx=0, gear_down=False)
                
        elif self.phase == FlightPhase.DEPARTURE:
            # 基于进度和高度：超过5%进度且高度>2000m
            if progress_ratio > self.departure_distance_ratio and alt > 2000:
                self.phase = FlightPhase.CLIMB
                
        elif self.phase == FlightPhase.CLIMB:
            # 基于进度和高度：超过爬升占比且接近巡航高度
            if progress_ratio > self.climb_distance_ratio or alt >= self.cruise_alt_m * 0.95:
                self.phase = FlightPhase.CRUISE
                
        elif self.phase == FlightPhase.CRUISE:
            # 基于进度：到达下降点
            if progress_ratio >= self.descent_distance_ratio:
                self.phase = FlightPhase.DESCENT
                
        elif self.phase == FlightPhase.DESCENT:
            # 基于进度：到达进近距离比例
            if progress_ratio >= self.approach_distance_ratio:
                self.phase = FlightPhase.APPROACH_DESCENT
                
        elif self.phase == FlightPhase.APPROACH_DESCENT:
            # 使用可配置的下滑道角度计算期望高度
            glide_slope_tan = np.tan(np.radians(self.glide_slope_angle))
            expected_glide_slope_alt = dist_to_dest_2d * glide_slope_tan
            
            # 进入FINAL的条件（使用可配置参数）：
            # 1. 距离小于阈值且高度接近期望下滑道高度
            # 2. 备用条件：距离<5km且高度<300m
            if (dist_to_dest_2d < self.final_entry_dist_m and 
                alt < expected_glide_slope_alt + self.final_entry_alt_margin) or \
               (dist_to_dest_2d < 5000 and alt < 300):
                self.phase = FlightPhase.FINAL
                self.model.set_config(flaps_idx=2, gear_down=True)  # 着陆形态
                if hasattr(self, '_final_prev_speed'):
                    del self._final_prev_speed
                
        elif self.phase == FlightPhase.DOWNWIND:
            if self.base_turn_point:
                dist_to_base = NavUtils.haversine_distance(
                    state['lat'], state['lon'],
                    self.base_turn_point[0], self.base_turn_point[1]
                )
                # 当接近四边转弯点时（3km内）
                if dist_to_base < 3000:
                    self.phase = FlightPhase.BASE
                    
        elif self.phase == FlightPhase.BASE:
            if self.final_turn_point:
                dist_to_final = NavUtils.haversine_distance(
                    state['lat'], state['lon'],
                    self.final_turn_point[0], self.final_turn_point[1]
                )
                # 进入五边：接近五边转弯点（2km内）
                if dist_to_final < 2000:
                    self.phase = FlightPhase.FINAL
                    self.model.set_config(flaps_idx=2, gear_down=True)  # 着陆形态
                    if hasattr(self, '_prev_target_speed'):
                        del self._prev_target_speed
                    if hasattr(self, '_final_prev_speed'):
                        del self._final_prev_speed
                    
        elif self.phase == FlightPhase.FINAL:
            # FINAL阶段应该很短，只用于最后接地
            # 更早触发FLARE
            if dist_to_dest_2d < 800 and alt < 60:
                self.phase = FlightPhase.FLARE
            elif alt < 25:  # 或者高度很低了
                self.phase = FlightPhase.FLARE
                
        elif self.phase == FlightPhase.FLARE:
            if alt < 2:
                self.phase = FlightPhase.TOUCHDOWN
                
        elif self.phase == FlightPhase.TOUCHDOWN:
            if tas < 30:
                self.phase = FlightPhase.ROLLOUT
    
    def _control_taxi(self) -> Tuple[float, float, float]:
        """滑行控制 - 加速到起飞速度"""
        # 地面滑行加速
        return 0.3, 0.0, 0.0  # 30%油门开始滑行
    
    def _control_takeoff_roll(self) -> Tuple[float, float, float]:
        """起飞滑跑控制"""
        return 1.0, 0.0, 0.0  # 全油门，保持跑道方向
    
    def _control_rotation(self) -> Tuple[float, float, float]:
        """抬轮控制"""
        return 1.0, 12.0, 0.0  # 全油门，抬头12度
    
    def _control_initial_climb(self) -> Tuple[float, float, float]:
        """初始爬升控制"""
        state = self.model.get_state()
        self.target_alt = 1500  # 爬升到5000英尺（约1500米）
        self.target_speed = self.approach_speed_ms * 1.4
        
        throttle, pitch, roll = self._pid_control(state)
        return throttle, min(pitch, 15.0), roll
    
    def _control_climb(self) -> Tuple[float, float, float]:
        """爬升控制"""
        state = self.model.get_state()
        self.target_alt = self.cruise_alt_m
        self.target_speed = self.cruise_speed_ms * 0.85
        
        return self._pid_control(state)
    
    def _control_cruise(self) -> Tuple[float, float, float]:
        """巡航控制"""
        state = self.model.get_state()
        self.target_alt = self.cruise_alt_m
        self.target_speed = self.cruise_speed_ms
        
        return self._pid_control(state)
    
    def _control_descent(self) -> Tuple[float, float, float]:
        """下降控制（基于航线占比的平滑下降）"""
        state = self.model.get_state()
        
        # 计算到目的地距离
        dist_to_dest = NavUtils.haversine_distance(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        # 计算航线进度
        dist_flown = self.total_route_distance - dist_to_dest
        progress_ratio = dist_flown / self.total_route_distance if self.total_route_distance > 0 else 0
        progress_ratio = np.clip(progress_ratio, 0.0, 1.0)
        
        # 下降阶段占比范围: descent_distance_ratio -> approach_distance_ratio
        # 例如: 0.80 -> 0.95 (即 80%-95% 这15%的航程用于下降)
        descent_start_ratio = self.descent_distance_ratio
        descent_end_ratio = self.approach_distance_ratio
        
        # 计算下降进度 (0.0 到 1.0)
        if progress_ratio <= descent_start_ratio:
            descent_progress = 0.0
        elif progress_ratio >= descent_end_ratio:
            descent_progress = 1.0
        else:
            descent_progress = (progress_ratio - descent_start_ratio) / (descent_end_ratio - descent_start_ratio)
        
        # 线性插值计算目标高度: 从巡航高度下降到五边高度
        self.target_alt = self.cruise_alt_m - \
                         (self.cruise_alt_m - self.pattern_alt_m) * descent_progress
        self.target_alt = max(self.pattern_alt_m, self.target_alt)
        
        # 速度控制：下降时适度减速
        self.target_speed = self.cruise_speed_ms * 0.8
        
        throttle, pitch, roll = self._pid_control(state)
        
        # 限制油门以确保下降
        throttle = min(throttle, 0.4)
        
        return throttle, pitch, roll
    
    def _control_approach_descent(self) -> Tuple[float, float, float]:
        """进近下降控制（下降到合适的进近高度和速度）"""
        state = self.model.get_state()
        
        # 计算到机场距离
        dist_to_airport = NavUtils.haversine_distance(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        # 目标：使用可配置下滑道角度逐渐降到合适的进近高度
        glide_slope_tan = np.tan(np.radians(self.glide_slope_angle))
        glide_slope_alt = max(200, dist_to_airport * glide_slope_tan)
        self.target_alt = min(self.pattern_alt_m, glide_slope_alt)
        
        # 速度：使用可配置的进近速度系数
        current_speed = state['tas']
        target_approach_speed = self.approach_speed_ms * self.approach_speed_factor
        
        # 初始化速度记录
        if not hasattr(self, '_prev_approach_target_speed'):
            self._prev_approach_target_speed = current_speed
        
        # 使用可配置的减速率平滑速度变化
        max_speed_change = self.speed_decel_rate
        speed_diff = target_approach_speed - self._prev_approach_target_speed
        
        if abs(speed_diff) > max_speed_change:
            self.target_speed = self._prev_approach_target_speed + np.sign(speed_diff) * max_speed_change
        else:
            self.target_speed = target_approach_speed
        
        self._prev_approach_target_speed = self.target_speed
        
        # 飞向机场
        self.target_heading = NavUtils.calculate_bearing(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        throttle, pitch, roll = self._pid_control(state)
        
        # 根据当前高度与目标高度的差距，动态调整油门
        alt_error = state['alt'] - self.target_alt
        if alt_error > 500:
            throttle = np.clip(throttle, 0.10, 0.40)
        elif alt_error > 200:
            throttle = np.clip(throttle, 0.15, 0.50)
        else:
            throttle = np.clip(throttle, 0.25, 0.65)
        
        return throttle, pitch, roll
    
    def _control_downwind(self) -> Tuple[float, float, float]:
        """三边（下风边）控制 - 平行跑道反向飞行，逐渐减速"""
        state = self.model.get_state()
        
        # 保持pattern高度
        self.target_alt = self.pattern_alt_m
        
        # 减速到进近速度的1.3倍
        self.target_speed = self.approach_speed_ms * 1.3
        
        # 飞向四边转弯点
        if self.base_turn_point:
            self.target_heading = NavUtils.calculate_bearing(
                state['lat'], state['lon'],
                self.base_turn_point[0], self.base_turn_point[1]
            )
        
        throttle, pitch, roll = self._pid_control(state)
        
        # 适度限制油门，帮助减速
        throttle = np.clip(throttle, 0.2, 0.7)
        
        return throttle, pitch, roll
    
    def _control_base(self) -> Tuple[float, float, float]:
        """四边（基线边）控制 - 继续下降和减速"""
        state = self.model.get_state()
        
        # 目标：下降到五边起始高度（约150m，500英尺）
        self.target_alt = self.final_approach_alt_m
        
        # 继续减速到进近速度的1.15倍
        self.target_speed = self.approach_speed_ms * 1.15
        
        # 飞向五边转弯点
        if self.final_turn_point:
            self.target_heading = NavUtils.calculate_bearing(
                state['lat'], state['lon'],
                self.final_turn_point[0], self.final_turn_point[1]
            )
        
        throttle, pitch, roll = self._pid_control(state)
        
        # 限制油门确保下降
        throttle = np.clip(throttle, 0.15, 0.6)
        
        return throttle, pitch, roll
    
    def _control_final(self) -> Tuple[float, float, float]:
        """
        五边（最后进近）控制 - 直接进近到跑道
        使用可配置参数实现平稳降落
        """
        state = self.model.get_state()
        
        # 计算距离跑道
        dist_to_runway = NavUtils.haversine_distance(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        # 始终飞向机场
        self.target_heading = NavUtils.calculate_bearing(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        # 速度：使用可配置的五边速度系数平滑减速
        current_speed = state['tas']
        
        if not hasattr(self, '_final_prev_speed'):
            self._final_prev_speed = current_speed
        
        # 目标速度：使用可配置的五边速度系数
        desired_speed = self.approach_speed_ms * self.final_speed_factor
        
        # 使用可配置的减速率（五边阶段用较慢减速率确保平稳）
        max_speed_change = self.speed_decel_rate * 0.8
        speed_diff = desired_speed - self._final_prev_speed
        
        if abs(speed_diff) > max_speed_change:
            self.target_speed = self._final_prev_speed + np.sign(speed_diff) * max_speed_change
        else:
            self.target_speed = desired_speed
        
        self._final_prev_speed = self.target_speed
        
        # 高度：使用可配置的下滑道角度
        glide_slope_tan = np.tan(np.radians(self.glide_slope_angle))
        self.target_alt = max(10, dist_to_runway * glide_slope_tan)
        
        throttle, pitch, roll = self._pid_control(state)
        
        # 根据距离动态调整油门范围确保平稳降落
        if dist_to_runway > 5000:
            throttle = np.clip(throttle, 0.15, 0.55)
        elif dist_to_runway > 2000:
            throttle = np.clip(throttle, 0.10, 0.50)
        else:
            throttle = np.clip(throttle, 0.05, 0.45)
        
        return throttle, pitch, roll
    
    def _control_flare(self) -> Tuple[float, float, float]:
        """拉平控制"""
        state = self.model.get_state()
        # 减小下降率，准备接地
        throttle = 0.0
        
        # 改进拉平逻辑：
        # 根据高度逐渐增加抬头
        # 目标：接地时 pitch 约 2-4 度，VS 约 -0.5 ~ -1.5 m/s
        
        current_pitch = state['pitch']
        current_vs = state['speed_v']
        
        # 如果垂直速度下降太快，增加抬头
        if current_vs < -3.0:
            target_pitch_cmd = current_pitch + 1.0
        elif current_vs < -1.0:
            target_pitch_cmd = current_pitch + 0.5
        elif current_vs > -0.2: # 飘飞了 (下降率太小)
             target_pitch_cmd = current_pitch - 0.5 # 稍微低头
        else:
            target_pitch_cmd = current_pitch # 保持
            
        # 限制拉平姿态
        pitch = np.clip(target_pitch_cmd, 0.0, 5.0)
        roll = 0.0
        return throttle, pitch, roll
    
    def _control_landing(self) -> Tuple[float, float, float]:
        """着陆滑跑控制"""
        state = self.model.get_state()
        self.model.set_config(flaps_idx=2, gear_down=True, spoilers=True)
        return 0.0, -2.0, 0.0  # 反推，压机头
    
    def _pid_control(self, state: dict) -> Tuple[float, float, float]:
        """
        PID控制器（改进版 - 增强稳定性和平滑性）
        
        Returns:
            (throttle, pitch, roll)
        """
        # 高度控制 - 采用能量方法避免振荡
        alt_err = self.target_alt - state['alt']
        
        # 根据高度误差计算目标垂直速度
        if alt_err > 0:  # 需要爬升
            desired_vs = np.clip(alt_err * 0.15, 0.0, 12.0)  # 限制最大爬升率12m/s
        else:  # 需要下降
            # FINAL阶段极其aggressive的下降
            if self.phase == FlightPhase.FINAL:
                # 进一步提高下降增益和下降率限制
                desired_vs = np.clip(alt_err * 0.5, -25.0, 0.0)  # 最大下降率25m/s
            else:
                desired_vs = np.clip(alt_err * 0.3, -15.0, 0.0)  # 最大下降率15m/s
        
        # 速度控制 - 根据阶段动态调整
        spd_err = self.target_speed - state['tas']
        
        # 根据飞行阶段调整速度控制增益
        if self.phase in [FlightPhase.DOWNWIND, FlightPhase.BASE]:
            # 五边航迹：使用较小增益保证平滑
            speed_gain = 0.005
        elif self.phase == FlightPhase.FINAL:
            # FINAL阶段使用适中的增益
            speed_gain = 0.006
        elif self.phase == FlightPhase.APPROACH_DESCENT:
            # APPROACH_DESCENT阶段也降低增益
            speed_gain = 0.004
        else:
            speed_gain = 0.01  # 正常增益
        
        throttle = np.clip(0.7 + spd_err * speed_gain, 0.0, 1.0)
        
        # 俯仰/航迹角控制 - 使用目标航迹角而非直接控制俯仰
        current_vs = state['speed_v']
        vs_err = desired_vs - current_vs
        
        # 计算目标航迹角（带阻尼）
        if state['tas'] > 30:
            target_gamma_deg = np.degrees(np.arcsin(np.clip(desired_vs / state['tas'], -0.35, 0.3)))
        else:
            target_gamma_deg = 0.0
        
        current_gamma_deg = np.degrees(state['gamma'])
        gamma_err = target_gamma_deg - current_gamma_deg
        
        # PD控制：比例 + 微分阻尼
        # FINAL阶段使用更高的增益以快速下降
        if self.phase == FlightPhase.FINAL:
            gamma_cmd = current_gamma_deg + gamma_err * 0.6  # 提高增益
        else:
            gamma_cmd = current_gamma_deg + gamma_err * 0.4
        gamma_cmd = np.clip(gamma_cmd, -10.0, 12.0)  # 扩大下降角度范围
        
        # 俯仰角 = 航迹角 + 期望攻角
        # 在爬升时使用较小的攻角，巡航/下降时使用适中攻角
        if desired_vs > 5:  # 爬升
            target_alpha = 8.0  # 8度攻角
        elif desired_vs < -5:  # 快速下降
            target_alpha = 2.0  # 降低攻角以加速下降
        elif desired_vs < -2:  # 正常下降
            target_alpha = 3.0
        else:  # 平飞/缓升
            target_alpha = 5.0
        
        # 失速保护 (Alpha Protection)
        max_alpha = 12.0
        current_alpha = state['alpha']
        if current_alpha > max_alpha:
            target_alpha = max_alpha * 0.8
            
        pitch = gamma_cmd + target_alpha
        pitch = np.clip(pitch, -15.0, 15.0)  # 限制俯仰角范围
        
        # 航向控制
        if self.current_wp_idx < len(self.waypoints):
            target_wp = self.waypoints[self.current_wp_idx]
            dist_to_wp = NavUtils.haversine_distance(
                state['lat'], state['lon'], target_wp[0], target_wp[1]
            )
            
            self.target_heading = NavUtils.calculate_bearing(
                state['lat'], state['lon'], target_wp[0], target_wp[1]
            )
            
            # 航路点切换（10km）
            if dist_to_wp < 10000:
                self.current_wp_idx += 1
        
        # 滚转控制
        hdg_err = self.target_heading - state['heading']
        if hdg_err > 180:
            hdg_err -= 360
        if hdg_err < -180:
            hdg_err += 360
        roll = np.clip(hdg_err * self.roll_gain, -self.max_roll, self.max_roll)
        
        return throttle, pitch, roll


class ShortHaulAutopilot(StandardAutopilot):
    """
    短程机自动驾驶系统
    适用于：ERJ-145, CRJ-900, E190, A220-300 等支线客机
    
    特点：
    - 快速爬升/下降（航程短，需要高效利用时间）
    - 更灵敏的操控响应
    - 较短的巡航段
    """
    
    def _init_tuning_params(self):
        """短程机调参 - 快速响应"""
        # 飞行阶段占比
        self.departure_distance_ratio = 0.08
        self.climb_distance_ratio = 0.20      # 更快到达巡航
        self.descent_distance_ratio = 0.70    # 更早开始下降
        self.approach_distance_ratio = 0.88   # 更早进入进近
        
        # 更高的爬升/下降率
        self.climb_rate_gain = 0.20
        self.descent_rate_gain = 0.40
        self.max_climb_vs = 15.0
        self.max_descent_vs = 18.0
        self.roll_gain = 2.0
        self.max_roll = 35.0
        
        # 短程机进近参数 - 较陡的下滑道，快速减速
        self.glide_slope_angle = 3.5          # 稍陡的下滑道
        self.final_entry_dist_m = 8000        # 较早进入FINAL
        self.final_entry_alt_margin = 60
        self.speed_decel_rate = 1.8           # 快速减速
        self.approach_speed_factor = 1.20
        self.final_speed_factor = 1.03


class MediumHaulAutopilot(StandardAutopilot):
    """
    中程机自动驾驶系统
    适用于：A320, A321, B737系列, B757 等窄体干线客机
    
    特点：
    - 标准爬升/下降率
    - 均衡的巡航段
    - 标准操控响应
    """
    
    def _init_tuning_params(self):
        """中程机调参 - 均衡配置"""
        # 飞行阶段占比
        self.departure_distance_ratio = 0.05
        self.climb_distance_ratio = 0.15
        self.descent_distance_ratio = 0.80
        self.approach_distance_ratio = 0.93   # 较早进入进近
        
        # 标准参数
        self.climb_rate_gain = 0.15
        self.descent_rate_gain = 0.30
        self.max_climb_vs = 12.0
        self.max_descent_vs = 15.0
        self.roll_gain = 1.5
        self.max_roll = 30.0
        
        # 中程机进近参数 - 标准配置
        self.glide_slope_angle = 3.0
        self.final_entry_dist_m = 10000
        self.final_entry_alt_margin = 80
        self.speed_decel_rate = 1.2
        self.approach_speed_factor = 1.25
        self.final_speed_factor = 1.05


class LongHaulAutopilot(StandardAutopilot):
    """
    长程机自动驾驶系统
    适用于：A330, A350, A380, B767, B777, B787, B747 等宽体客机
    
    特点：
    - 平缓爬升/下降（乘客舒适性，燃油效率）
    - 较长的巡航段
    - 平稳的操控响应
    """
    
    def _init_tuning_params(self):
        """长程机调参 - 平稳舒适"""
        # 飞行阶段占比
        self.departure_distance_ratio = 0.03
        self.climb_distance_ratio = 0.10      # 较长爬升段
        self.descent_distance_ratio = 0.85    # 较晚开始下降
        self.approach_distance_ratio = 0.92   # 更早进入进近（给大飞机更多稳定时间）
        
        # 更平缓的参数
        self.climb_rate_gain = 0.12
        self.descent_rate_gain = 0.30        # 略微增加增益
        self.max_climb_vs = 10.0
        self.max_descent_vs = 25.0           # 大幅增加允许的下降率，确保可以追赶下滑道
        self.roll_gain = 1.2
        self.max_roll = 25.0
        
        # 长程机进近参数 - 较缓的下滑道，平稳减速
        self.glide_slope_angle = 2.8          # 较缓的下滑道
        self.final_entry_dist_m = 11000       # 缩短进入FINAL距离 (11km)
        self.final_entry_alt_margin = 100
        self.speed_decel_rate = 1.0           # 增加减速率以适应高能进近
        self.approach_speed_factor = 1.30
        self.final_speed_factor = 1.08

    def _control_final(self) -> Tuple[float, float, float]:
        """
        覆盖五边控制 logic
        增强对准跑道的稳定性，防止五边S形机动
        """
        state = self.model.get_state()
        
        # 计算距离跑道
        dist_to_runway = NavUtils.haversine_distance(
            state['lat'], state['lon'],
            self.destination_airport[0], self.destination_airport[1]
        )
        
        # 1. 航向控制增强
        # 计算直飞机场的方位
        bearing_to_airport = NavUtils.calculate_bearing(
             state['lat'], state['lon'],
             self.destination_airport[0], self.destination_airport[1]
        )
        
        # 如果距离很近 (<3km)，强行混合跑道方向以稳定姿态
        if dist_to_runway < 3000:
            # 权重随距离变化：越近越依赖跑道方向
            # 3km处 0%, 0km处 80% (保留20%修正误差)
            runway_weight = 0.8 * (1.0 - dist_to_runway / 3000)
            
            # 角度混合
            hdg_diff = self.runway_heading - bearing_to_airport
            if hdg_diff > 180: hdg_diff -= 360
            elif hdg_diff < -180: hdg_diff += 360
            
            self.target_heading = bearing_to_airport + hdg_diff * runway_weight
        else:
             self.target_heading = bearing_to_airport
             
        # 2. 速度控制
        # 继承父类的减速逻辑，但确保目标不低于Vref
        current_speed = state['tas']
        if not hasattr(self, '_final_prev_speed'):
            self._final_prev_speed = current_speed
            
        desired_speed = self.approach_speed_ms * self.final_speed_factor
        max_speed_change = self.speed_decel_rate
        speed_diff = desired_speed - self._final_prev_speed
        
        if abs(speed_diff) > max_speed_change:
            self.target_speed = self._final_prev_speed + np.sign(speed_diff) * max_speed_change
        else:
            self.target_speed = desired_speed
        
        self._final_prev_speed = self.target_speed
        
        # 3. 高度控制
        glide_slope_tan = np.tan(np.radians(self.glide_slope_angle))
        self.target_alt = max(10, dist_to_runway * glide_slope_tan)
        
        # 4. PID控制
        throttle, pitch, roll = self._pid_control(state)
        
        # 5. 提前Flare检测
        # 如果高度很低且距离很近，强制进入FLARE (比标准版更早)
        if (dist_to_runway < 2000 and state['alt'] < 60) or state['alt'] < 30:
            self.phase = FlightPhase.FLARE
            
        return throttle, pitch, roll

    def _pid_control(self, state: dict) -> Tuple[float, float, float]:
        """
        重写PID控制，增加能量管理逻辑和动态基础油门
        """
        # 动态调整基础油门 (Base Throttle)
        # 这一步非常关键：标准版使用固定的0.7，导致大飞机无法减速
        if self.phase in [FlightPhase.DESCENT, FlightPhase.APPROACH_DESCENT]:
            base_throttle = 0.3
        elif self.phase in [FlightPhase.DOWNWIND, FlightPhase.BASE]:
            base_throttle = 0.45
        elif self.phase == FlightPhase.FINAL:
            base_throttle = 0.45 # 保持一定动力
        else:
            base_throttle = 0.7
            
        # 复制父类PID逻辑的主要部分，但替换base_throttle
        # 为避免完全复制大段代码，这里我们通过调整返回的throttle来"hack"一下
        # 原始公式: throttle = clip(0.7 + err * gain, 0, 1)
        # 我们想变成: throttle' = clip(base + err * gain, 0, 1)
        # 近似: throttle' = throttle - 0.7 + base
        
        throttle_orig, pitch, roll = super()._pid_control(state)
        
        # 修正油门
        throttle_corrected = throttle_orig - 0.7 + base_throttle
        throttle = np.clip(throttle_corrected, 0.0, 1.0)
        
        # 能量管理增强 logic (保持之前的代码)
        if self.phase in [FlightPhase.DESCENT, FlightPhase.APPROACH_DESCENT]:
            # 计算能量误差
            spd_err = self.target_speed - state['tas']
            alt_err = self.target_alt - state['alt']
            
            # 如果速度过快或者高度过高
            if spd_err < -5.0 or alt_err < -100:
                # 强制减小油门
                throttle_limit = 0.1 if spd_err > -10 else 0.0
                throttle = min(throttle, throttle_limit)
                
                # 自动扰流板逻辑 (Speed Brakes)
                if spd_err < -15.0 or (alt_err < -300 and self.phase == FlightPhase.DESCENT):
                    if throttle < 0.05:
                        self.model.set_config(flaps_idx=state['flaps'], gear_down=state['gear'], spoilers=True)
                else:
                    if state['spoilers'] and self.phase != FlightPhase.ROLLOUT:
                         self.model.set_config(flaps_idx=state['flaps'], gear_down=state['gear'], spoilers=False)
        
        return throttle, pitch, roll

    def _calculate_point_from_bearing_dist(self, lat1, lon1, bearing, distance_m):
        """Helper to calculate new coordinate"""
        R = 6371000.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        bearing_rad = np.radians(bearing)
        
        lat2_rad = np.arcsin(np.sin(lat1_rad) * np.cos(distance_m/R) + 
                             np.cos(lat1_rad) * np.sin(distance_m/R) * np.cos(bearing_rad))
        
        lon2_rad = lon1_rad + np.arctan2(np.sin(bearing_rad) * np.sin(distance_m/R) * np.cos(lat1_rad),
                                         np.cos(distance_m/R) - np.sin(lat1_rad) * np.sin(lat2_rad))
        
        return np.degrees(lat2_rad), np.degrees(lon2_rad)

    def load_route(self, waypoints: List[Tuple[float, float]], 
                   departure_alt: float = 10.0,
                   runway_heading: float = 0.0):
        """
        覆盖加载航路点方法
        1. 实现动态下降点(TOD)计算
        2. [NEW] 实现智能进近点注入(Smart Approach Injection)
        """
        # ==========================================
        # Smart Approach Injection Logic
        # ==========================================
        # Check if we have a valid runway heading to align with
        if runway_heading is not None and len(waypoints) >= 2:
            # Use provided heading, or if 0.0 (default), maybe we just skip or try to infer?
            # For now, we rely on the passed runway_heading being accurate.
            
            if runway_heading != 0.0:
                final_wp = waypoints[-1]
                prev_wp = waypoints[-2]
                
                # Calculate current approach bearing
                approach_bearing = NavUtils.calculate_bearing(
                    prev_wp[0], prev_wp[1],
                    final_wp[0], final_wp[1]
                )
                
                # Calculate difference angle (shortest path)
                diff = abs(approach_bearing - runway_heading)
                if diff > 180: diff = 360 - diff
                
                # If angle is too steep (> 30 degrees), inject a fix
                if diff > 30.0:
                    print(f"  [Smart Autopilot] Bad Approach detected (Diff {diff:.1f}°). Injecting Fix.")
                    
                    # Calculate fix position: 60km out, opposite to runway heading
                    fix_bearing = (runway_heading + 180) % 360
                    fix_dist = 60000.0 # 60km
                    
                    fix_lat, fix_lon = self._calculate_point_from_bearing_dist(
                        final_wp[0], final_wp[1], fix_bearing, fix_dist
                    )
                    
                    # Insert fix before destination
                    new_waypoints = list(waypoints)
                    new_waypoints.insert(-1, (fix_lat, fix_lon))
                    waypoints = new_waypoints
                    print(f"  [Smart Autopilot] Injected Approach Fix at {fix_lat:.4f}, {fix_lon:.4f}")

        super().load_route(waypoints, departure_alt, runway_heading)
        
        # 重新计算下降和进近阶段的比例
        # 典型的民航客机下降剖面约为3度下滑道
        # 需要下降的高度
        descent_height_m = self.cruise_alt_m - self.pattern_alt_m
        if descent_height_m <= 0:
            return
            
        # 3度下滑道所需的直线距离
        # descent_dist = height / tan(3deg)
        # tan(3deg) ≈ 0.0524
        glide_angle_rad = np.radians(3.0)
        descent_distance_m = descent_height_m / np.tan(glide_angle_rad)
        
        # 增加一些缓冲距离用于平飞减速 (约30-50km)
        decel_buffer_m = 40000 
        total_descent_dist_m = descent_distance_m + decel_buffer_m
        
        # 打印调试信息
        print(f"Long-Haul TOD Calc: Dist={self.total_route_distance/1000:.1f}km, CruiseAlt={self.cruise_alt_m:.0f}m")
        print(f"  DescentDist={total_descent_dist_m/1000:.1f}km (Profile={descent_distance_m/1000:.1f}km + Buffer={decel_buffer_m/1000:.1f}km)")
        
        # 如果航程极短，使用默认比例
        if self.total_route_distance < total_descent_dist_m * 1.5:
            return
            
        # 计算新的下降点比例
        # 下降点 = 总距离 - 下降所需距离
        dist_to_tod = self.total_route_distance - total_descent_dist_m
        
        # 更新比例
        # 确保不早于爬升结束
        min_descent_ratio = self.climb_distance_ratio + 0.05
        new_descent_ratio = max(min_descent_ratio, dist_to_tod / self.total_route_distance)
        
        self.descent_distance_ratio = new_descent_ratio
        
        # 进近点设为最后30km左右
        approach_dist_m = 30000
        new_approach_ratio = max(new_descent_ratio + 0.01, 
                                (self.total_route_distance - approach_dist_m) / self.total_route_distance)
        self.approach_distance_ratio = new_approach_ratio
        
        print(f"  New Descent Ratio: {self.descent_distance_ratio:.4f} (was 0.85)")
        print(f"  New Approach Ratio: {self.approach_distance_ratio:.4f} (was 0.92)")


def create_autopilot(model: SixDOFModel, 
                     range_category: str = None,
                     cruise_speed_mach: Optional[float] = None) -> BaseAutopilot:
    """
    根据航程分类创建合适的自动驾驶实例
    
    Args:
        model: 六自由度飞机模型
        range_category: 航程分类 ('SHORT_HAUL', 'MEDIUM_HAUL', 'LONG_HAUL')
                       如果为None，自动从模型推断
        cruise_speed_mach: 可选的巡航马赫数
    
    Returns:
        合适的自动驾驶实例
    """
    # 如果未指定分类，从飞机航程推断
    if range_category is None:
        max_range = model.params.performance.max_range_km
        if max_range < 4000:
            range_category = 'SHORT_HAUL'
        elif max_range < 10000:
            range_category = 'MEDIUM_HAUL'
        else:
            range_category = 'LONG_HAUL'
    
    # 根据分类创建实例
    if range_category == 'SHORT_HAUL':
        return ShortHaulAutopilot(model, cruise_speed_mach)
    elif range_category == 'LONG_HAUL':
        return LongHaulAutopilot(model, cruise_speed_mach)
    else:  # MEDIUM_HAUL or default
        return MediumHaulAutopilot(model, cruise_speed_mach)

