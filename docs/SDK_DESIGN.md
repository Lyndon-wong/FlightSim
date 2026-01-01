# FlightSim SDK 设计方案

为 FlightSim 飞行轨迹仿真库设计一套**易用、功能深入、层次分明**的 Python SDK 接口。

---

## 一、设计理念

| 原则 | 说明 |
|------|------|
| **渐进式复杂度** | 新手一行代码即可生成航迹，专家可深入控制每个仿真步 |
| **合理默认值** | 未指定参数时使用机型最佳配置，可复现 |
| **类型提示完备** | 全面的 Type Hints，方便 IDE 补全和静态检查 |
| **文档驱动** | 每个公开 API 都有清晰的中文文档字符串 |

---

## 二、API 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                      高层 API (Quick API)                    │
│   generate_trajectory() / list_aircraft() / list_routes()   │
│                  一行代码完成完整飞行仿真                      │
├─────────────────────────────────────────────────────────────┤
│                      中层 API (Flight API)                   │
│           Aircraft.fly() / Trajectory / Route               │
│            创建飞机对象，可定制飞行参数                         │
├─────────────────────────────────────────────────────────────┤
│                      底层 API (Core API)                     │
│     DynamicsModel / Autopilot / AircraftParams / NavUtils   │
│           直接控制六自由度模型，手动仿真每一步                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、高层 API 设计 (Quick API)

### 3.1 `generate_trajectory()` - 一键生成航迹

```python
def generate_trajectory(
    aircraft: str,
    origin: Union[str, Tuple[float, float]],
    destination: Union[str, Tuple[float, float]],
    *,
    # ===== 航线配置 =====
    route: Optional[str] = None,           # 预设航线名 (如 "PEK-LAX")
    waypoints: Optional[List[Tuple[float, float]]] = None,  # 自定义中间航路点
    
    # ===== 巡航配置 =====
    cruise_altitude_ft: Optional[float] = None,   # 巡航高度 (英尺)，默认使用机型推荐值
    cruise_speed_mach: Optional[float] = None,    # 巡航马赫数，默认使用机型推荐值
    
    # ===== 仿真配置 =====
    time_step: float = 0.5,               # 仿真时间步长 (秒)
    max_time: float = 43200,              # 最大仿真时间 (秒，默认12小时)
    
    # ===== 输出配置 =====
    output_format: Literal["dataframe", "dict", "csv"] = "dataframe",
    output_file: Optional[str] = None,    # 保存到文件路径
    
    # ===== 高级选项 =====
    random_seed: Optional[int] = None,    # 随机种子，设置后可复现
    noise_level: float = 0.0,             # 噪声级别 0.0-1.0
    include_detailed_forces: bool = False # 是否输出详细力学数据
) -> Union[pd.DataFrame, Dict, str]:
    """
    一键生成完整飞行轨迹
    
    参数:
        aircraft: 飞机型号，如 "B737-800", "A320-200"
        origin: 起点，可以是机场代码 "PEK" 或坐标元组 (40.08, 116.58)
        destination: 终点，格式同起点
        route: 使用预设航线名称（会自动查找 waypoints.csv）
        waypoints: 自定义中间航路点列表
        cruise_altitude_ft: 巡航高度，None 时使用机型推荐值
        cruise_speed_mach: 巡航马赫数，None 时使用机型推荐值
        time_step: 仿真步长（秒）
        max_time: 最大仿真时间（秒）
        output_format: 输出格式
        output_file: 输出文件路径
        random_seed: 随机种子
        noise_level: 传感器噪声级别
        include_detailed_forces: 是否包含详细力学数据
    
    返回:
        根据 output_format 返回 DataFrame、字典或 CSV 文件路径
    
    示例:
        >>> import flightsim as fs
        >>> # 最简调用
        >>> df = fs.generate_trajectory("B737-800", "PEK", "PVG")
        >>> 
        >>> # 指定巡航参数
        >>> df = fs.generate_trajectory(
        ...     "A320-200", 
        ...     origin=(40.08, 116.58),
        ...     destination=(31.14, 121.81),
        ...     cruise_altitude_ft=35000,
        ...     cruise_speed_mach=0.78
        ... )
        >>> 
        >>> # 使用预设航线
        >>> df = fs.generate_trajectory("B777-300ER", "PEK", "LAX", route="北京-洛杉矶")
    """
```

### 3.2 `list_aircraft()` - 查询可用机型

```python
def list_aircraft(
    category: Optional[Literal["short_haul", "medium_haul", "long_haul", "all"]] = "all",
    manufacturer: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    获取可用飞机型号列表
    
    参数:
        category: 按航程类别筛选
        manufacturer: 按制造商筛选 ("Boeing", "Airbus", "Embraer" 等)
    
    返回:
        飞机型号信息列表，每项包含：
        - aircraft_type: 型号
        - manufacturer: 制造商
        - range_km: 航程 (km)
        - typical_pax: 典型座位数
        - cruise_speed_mach: 巡航马赫数
        - cruise_alt_ft: 巡航高度 (ft)
    
    示例:
        >>> fs.list_aircraft(category="long_haul")
        [
            {"aircraft_type": "B777-300ER", "manufacturer": "Boeing", ...},
            {"aircraft_type": "A350-900", "manufacturer": "Airbus", ...},
            ...
        ]
    """
```

### 3.3 `list_routes()` - 查询预设航线

```python
def list_routes(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    category: Optional[Literal["中国", "亚洲", "跨太平洋", "跨大西洋", "美国", "欧洲", "其他"]] = None
) -> List[Dict[str, Any]]:
    """
    获取预设航线列表
    
    参数:
        origin: 筛选起点机场代码
        destination: 筛选终点机场代码
        category: 筛选航线类别
    
    返回:
        航线信息列表，每项包含：
        - route_name: 航线名称
        - origin_code / destination_code: 起终点代码
        - distance_km: 航线距离
        - recommended_aircraft: 推荐机型
        - num_waypoints: 航路点数量
    
    示例:
        >>> fs.list_routes(origin="PEK")
        [
            {"route_name": "北京-上海", "origin_code": "PEK", "destination_code": "PVG", ...},
            {"route_name": "北京-洛杉矶", "origin_code": "PEK", "destination_code": "LAX", ...},
            ...
        ]
    """
```

### 3.4 `get_airport()` - 查询机场信息

```python
def get_airport(code: str) -> Optional[Dict[str, Any]]:
    """
    根据机场代码获取机场信息
    
    参数:
        code: IATA 机场代码 (如 "PEK", "LAX")
    
    返回:
        机场信息字典，包含：
        - code: 机场代码
        - name: 机场名称
        - city: 城市
        - country: 国家
        - lat: 纬度
        - lon: 经度
        - elevation_m: 海拔高度 (米)
        
        若机场不存在返回 None
    """
```

---

## 四、中层 API 设计 (Flight API)

### 4.1 `Aircraft` 类 - 飞机对象

```python
class Aircraft:
    """
    飞机类 - 封装飞机性能参数和飞行能力
    
    属性:
        aircraft_type: 飞机型号
        params: AircraftParams 对象（完整性能参数）
        dm: DynamicsModel 实例（底层动力学模型）
    """
    
    def __init__(
        self,
        aircraft_type: str,
        *,
        initial_fuel_kg: Optional[float] = None,  # 初始燃油量，None=满油
        payload_kg: Optional[float] = None        # 载荷重量，None=典型载荷
    ):
        """
        创建飞机实例
        
        参数:
            aircraft_type: 飞机型号 (如 "B737-800")
            initial_fuel_kg: 初始燃油量 (kg)，默认为满油
            payload_kg: 载荷重量 (kg)，默认为典型载荷
        
        示例:
            >>> plane = fs.Aircraft("A320-200")
            >>> plane = fs.Aircraft("B777-300ER", initial_fuel_kg=150000)
        """
    
    # ===== 核心方法 =====
    
    def fly(
        self,
        origin: Union[str, Tuple[float, float]],
        destination: Union[str, Tuple[float, float]],
        *,
        route: Optional[str] = None,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        cruise_altitude_ft: Optional[float] = None,
        cruise_speed_mach: Optional[float] = None,
        departure_runway_heading: Optional[float] = None,  # 起飞跑道方向
        arrival_runway_heading: Optional[float] = None,    # 着陆跑道方向
        time_step: float = 0.5,
        max_time: float = 43200,
        autopilot_mode: Literal["standard", "fuel_optimal", "time_optimal"] = "standard"
    ) -> "Trajectory":
        """
        执行完整飞行，返回轨迹对象
        
        参数:
            origin: 起点
            destination: 终点
            route: 预设航线名
            waypoints: 自定义航路点
            cruise_altitude_ft: 巡航高度
            cruise_speed_mach: 巡航速度
            departure_runway_heading: 起飞跑道方向 (度)
            arrival_runway_heading: 着陆跑道方向 (度)
            time_step: 仿真步长
            max_time: 最大仿真时间
            autopilot_mode: 自动驾驶模式
        
        返回:
            Trajectory 对象
        """
    
    def calculate_range(
        self,
        payload_kg: Optional[float] = None,
        fuel_kg: Optional[float] = None,
        cruise_altitude_ft: Optional[float] = None
    ) -> float:
        """
        计算当前配置下的航程 (km)
        """
    
    def calculate_flight_time(
        self,
        distance_km: float,
        cruise_altitude_ft: Optional[float] = None,
        cruise_speed_mach: Optional[float] = None
    ) -> float:
        """
        估算飞行时间 (秒)
        """
    
    def calculate_fuel_required(
        self,
        distance_km: float,
        payload_kg: Optional[float] = None
    ) -> float:
        """
        估算所需燃油量 (kg)
        """
    
    # ===== 属性 =====
    
    @property
    def specs(self) -> Dict[str, Any]:
        """飞机规格摘要"""
    
    @property
    def performance(self) -> Dict[str, Any]:
        """性能数据摘要"""
    
    @property 
    def dm(self) -> "DynamicsModel":
        """底层动力学模型（高级用户）"""
    
    def __repr__(self) -> str:
        """友好的字符串表示"""
        # Aircraft(B737-800): 79016kg MTOW, 5765km range, M0.789 cruise
```

### 4.2 `Trajectory` 类 - 轨迹对象

```python
class Trajectory:
    """
    飞行轨迹类 - 封装仿真结果和分析方法
    """
    
    # ===== 数据访问 =====
    
    @property
    def data(self) -> pd.DataFrame:
        """完整轨迹数据 DataFrame"""
    
    @property
    def summary(self) -> Dict[str, Any]:
        """
        飞行摘要信息:
        - total_time_s: 总飞行时间 (秒)
        - total_distance_km: 总航程 (km)
        - fuel_consumed_kg: 消耗燃油 (kg)
        - max_altitude_m: 最高高度 (米)
        - max_speed_mach: 最高马赫数
        - phase_durations: 各阶段时长字典
        """
    
    @property
    def phases(self) -> List[str]:
        """经历的飞行阶段列表"""
    
    # ===== 数据提取 =====
    
    def get_positions(self) -> np.ndarray:
        """返回 (N, 3) 数组: [lat, lon, alt]"""
    
    def get_velocities(self) -> np.ndarray:
        """返回 (N, 3) 数组: [vn, ve, vd] m/s"""
    
    def get_attitudes(self) -> np.ndarray:
        """返回 (N, 3) 数组: [pitch, roll, heading] 度"""
    
    def get_times(self) -> np.ndarray:
        """返回时间序列 (秒)"""
    
    def at_time(self, t: float) -> Dict[str, Any]:
        """获取指定时刻的状态（插值）"""
    
    def at_phase(self, phase: str) -> pd.DataFrame:
        """获取指定飞行阶段的数据"""
    
    # ===== 采样与重采样 =====
    
    def resample(
        self,
        interval: float = 1.0,
        method: Literal["linear", "cubic", "nearest"] = "linear"
    ) -> "Trajectory":
        """按指定时间间隔重采样"""
    
    def downsample(self, factor: int = 10) -> "Trajectory":
        """降采样"""
    
    # ===== 导出 =====
    
    def to_csv(self, path: str, **kwargs) -> str:
        """导出为 CSV 文件"""
    
    def to_json(self, path: str) -> str:
        """导出为 JSON 文件"""
    
    def to_geojson(self, path: str) -> str:
        """导出为 GeoJSON 格式（方便 GIS 工具使用）"""
    
    def to_kml(self, path: str) -> str:
        """导出为 KML 格式（Google Earth）"""
    
    # ===== 可视化 =====
    
    def plot_3d(self, **kwargs) -> "matplotlib.figure.Figure":
        """3D 轨迹图"""
    
    def plot_profile(self, **kwargs) -> "matplotlib.figure.Figure":
        """高度剖面图"""
    
    def plot_map(self, **kwargs) -> "folium.Map":
        """地图轨迹（使用 Folium）"""
    
    def plot_phases(self, **kwargs) -> "matplotlib.figure.Figure":
        """飞行阶段时间轴图"""
    
    def plot_energy(self, **kwargs) -> "matplotlib.figure.Figure":
        """能量变化图（动能、势能、总能量）"""
```

### 4.3 `Route` 类 - 航线对象

```python
class Route:
    """
    航线类 - 封装航路点和航线信息
    """
    
    def __init__(
        self,
        origin: Union[str, Tuple[float, float]],
        destination: Union[str, Tuple[float, float]],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        name: Optional[str] = None
    ):
        """
        创建航线
        
        参数:
            origin: 起点
            destination: 终点
            waypoints: 中间航路点列表
            name: 航线名称
        """
    
    @classmethod
    def from_preset(cls, route_name: str) -> "Route":
        """从预设航线创建"""
    
    @classmethod
    def from_airports(cls, origin: str, destination: str, auto_waypoints: bool = True) -> "Route":
        """从机场代码创建，可自动查找预设航路点"""
    
    @classmethod
    def great_circle(cls, origin: Tuple[float, float], destination: Tuple[float, float], num_points: int = 10) -> "Route":
        """生成大圆航线"""
    
    # ===== 属性 =====
    
    @property
    def distance_km(self) -> float:
        """航线总距离 (km)"""
    
    @property
    def points(self) -> List[Tuple[float, float]]:
        """所有航路点列表（含起终点）"""
    
    @property
    def bearings(self) -> List[float]:
        """各航段航向 (度)"""
    
    @property
    def segment_distances(self) -> List[float]:
        """各航段距离 (km)"""
    
    # ===== 方法 =====
    
    def add_waypoint(self, lat: float, lon: float, index: Optional[int] = None) -> "Route":
        """添加航路点"""
    
    def remove_waypoint(self, index: int) -> "Route":
        """移除航路点"""
    
    def reverse(self) -> "Route":
        """返航航线"""
    
    def to_geojson(self) -> Dict:
        """转换为 GeoJSON"""
```

---

## 五、底层 API 设计 (Core API)

### 5.1 `DynamicsModel` 类 - 六自由度动力学模型

```python
class DynamicsModel:
    """
    六自由度飞行动力学模型
    
    实现基于物理的飞行器运动仿真，包括：
    - 详细气动力计算（考虑马赫数、雷诺数效应）
    - 发动机推力模型（海拔和速度影响）
    - 燃油消耗计算
    - 大气模型（ISA 标准大气）
    """
    
    def __init__(
        self,
        aircraft_type: str,
        *,
        start_lat: float,
        start_lon: float,
        start_alt: float = 10.0,
        start_heading: float = 0.0,
        start_speed: float = 0.0,
        dt: float = 1.0,
        initial_fuel_kg: Optional[float] = None,
        payload_kg: Optional[float] = None
    ):
        """
        初始化动力学模型
        
        参数:
            aircraft_type: 飞机型号
            start_lat: 初始纬度 (度)
            start_lon: 初始经度 (度)
            start_alt: 初始海拔高度 (米)
            start_heading: 初始航向 (度，0=北)
            start_speed: 初始地速 (m/s)
            dt: 仿真时间步长 (秒)
            initial_fuel_kg: 初始燃油 (kg)
            payload_kg: 载荷重量 (kg)
        """
    
    # ==================== 状态读取 ====================
    
    @property
    def state(self) -> Dict[str, Any]:
        """
        获取当前完整状态
        
        返回字典包含：
            # 位置
            lat: float          - 纬度 (度)
            lon: float          - 经度 (度)
            alt: float          - 海拔高度 (米)
            
            # 速度
            tas: float          - 真空速 (m/s)
            ias: float          - 指示空速 (m/s)
            gs: float           - 地速 (m/s)
            mach: float         - 马赫数
            speed_v: float      - 垂直速度 (m/s，正为上升)
            speed_n: float      - 北向速度分量 (m/s)
            speed_e: float      - 东向速度分量 (m/s)
            
            # 姿态
            pitch: float        - 俯仰角 (度，正为抬头)
            roll: float         - 滚转角 (度，正为右倾)
            heading: float      - 航向 (度，0-360)
            gamma: float        - 航迹角/爬升角 (度)
            alpha: float        - 攻角 (度)
            
            # 力学
            lift: float         - 升力 (N)
            drag: float         - 阻力 (N)
            thrust: float       - 推力 (N)
            load_factor: float  - 载荷因子 (g)
            
            # 质量
            mass: float         - 当前总质量 (kg)
            fuel: float         - 剩余燃油 (kg)
            
            # 配置
            flaps: int          - 襟翼位置 (0/1/2)
            gear: bool          - 起落架状态
            spoilers: bool      - 扰流板状态
            
            # 时间
            time: float         - 仿真时间 (秒)
        """
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """当前位置 (lat, lon, alt)"""
    
    @property
    def velocity(self) -> Tuple[float, float, float]:
        """当前速度 (vn, ve, vd) m/s，NED 坐标系"""
    
    @property
    def attitude(self) -> Tuple[float, float, float]:
        """当前姿态 (pitch, roll, heading) 度"""
    
    @property
    def speed(self) -> float:
        """真空速 TAS (m/s)"""
    
    @property
    def mach(self) -> float:
        """马赫数"""
    
    @property
    def altitude(self) -> float:
        """海拔高度 (米)"""
    
    @property
    def fuel(self) -> float:
        """剩余燃油 (kg)"""
    
    @property
    def mass(self) -> float:
        """当前总质量 (kg)"""
    
    @property
    def time(self) -> float:
        """当前仿真时间 (秒)"""
    
    @property
    def aircraft_params(self) -> "AircraftParams":
        """飞机性能参数（只读）"""
    
    # ==================== 配置控制 ====================
    
    def set_config(
        self,
        *,
        flaps: Optional[int] = None,       # 0=收起, 1=起飞, 2=着陆
        gear_down: Optional[bool] = None,
        spoilers: Optional[bool] = None
    ) -> None:
        """
        设置飞行器气动配置
        
        参数:
            flaps: 襟翼位置索引
            gear_down: 起落架是否放下
            spoilers: 扰流板是否展开
        """
    
    def reset(
        self,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        alt: Optional[float] = None,
        heading: Optional[float] = None,
        speed: Optional[float] = None
    ) -> None:
        """重置状态到指定位置"""
    
    # ==================== 仿真推进 ====================
    
    def step(
        self,
        throttle: float,
        pitch_target: float,
        roll_target: float
    ) -> Dict[str, Any]:
        """
        执行一个仿真步长
        
        参数:
            throttle: 油门百分比 (0.0 - 1.0)
            pitch_target: 目标俯仰角 (度，-90 到 +90)
            roll_target: 目标滚转角 (度，-60 到 +60)
        
        返回:
            更新后的状态字典（同 state 属性格式）
        
        示例:
            >>> dm = fs.DynamicsModel("B737-800", start_lat=40.08, start_lon=116.58)
            >>> # 起飞滑跑
            >>> for i in range(100):
            ...     state = dm.step(throttle=1.0, pitch_target=0, roll_target=0)
            >>> # 抬头爬升
            >>> state = dm.step(throttle=0.95, pitch_target=15, roll_target=0)
        """
    
    def step_n(
        self,
        throttle: float,
        pitch_target: float,
        roll_target: float,
        n_steps: int
    ) -> List[Dict[str, Any]]:
        """执行多个仿真步，返回状态历史"""
    
    # ==================== 高级控制辅助 ====================
    
    def calculate_control_for_target(
        self,
        *,
        target_altitude: Optional[float] = None,    # 目标高度 (米)
        target_heading: Optional[float] = None,     # 目标航向 (度)
        target_speed: Optional[float] = None,       # 目标空速 (m/s)
        target_vertical_speed: Optional[float] = None,  # 目标垂直速度 (m/s)
        target_position: Optional[Tuple[float, float]] = None  # 目标位置 (lat, lon)
    ) -> Tuple[float, float, float]:
        """
        计算达到目标所需的控制输入
        
        返回:
            (throttle, pitch_target, roll_target) 元组
        
        示例:
            >>> # 计算爬升到 10000 米所需的控制
            >>> t, p, r = dm.calculate_control_for_target(target_altitude=10000)
            >>> dm.step(t, p, r)
        """
    
    def calculate_level_flight_throttle(self, target_speed: Optional[float] = None) -> float:
        """计算维持平飞所需油门"""
    
    def calculate_climb_rate(self, throttle: float = 1.0) -> float:
        """计算当前条件下的最大爬升率 (m/s)"""
    
    def calculate_descent_rate(self, throttle: float = 0.0, spoilers: bool = False) -> float:
        """计算下降率 (m/s)"""
    
    # ==================== 大气和性能查询 ====================
    
    def get_atmosphere(self, altitude: Optional[float] = None) -> Dict[str, float]:
        """
        获取大气参数
        
        参数:
            altitude: 高度 (米)，None 表示当前高度
        
        返回:
            {
                "density": float,      # 密度 (kg/m³)
                "temperature": float,  # 温度 (K)
                "pressure": float,     # 压力 (Pa)
                "sound_speed": float   # 音速 (m/s)
            }
        """
    
    def get_max_thrust(self) -> float:
        """当前条件下的最大可用推力 (N)"""
    
    def get_min_speed(self) -> float:
        """当前配置的最小飞行速度 / 失速速度 (m/s)"""
    
    def get_max_speed(self) -> float:
        """当前条件下的最大飞行速度 (m/s)"""
    
    def get_fuel_flow_rate(self, throttle: float = 1.0) -> float:
        """计算燃油流量 (kg/s)"""
    
    def get_range_remaining(self) -> float:
        """剩余燃油可飞行距离 (km)"""
    
    def get_endurance_remaining(self) -> float:
        """剩余燃油可飞行时间 (秒)"""
    
    # ==================== 气动力查询 ====================
    
    def get_lift_coefficient(self, alpha: Optional[float] = None) -> float:
        """获取升力系数 CL"""
    
    def get_drag_coefficient(self, alpha: Optional[float] = None) -> float:
        """获取阻力系数 CD"""
    
    def get_lift_to_drag_ratio(self) -> float:
        """获取升阻比 L/D"""
```

### 5.2 `Autopilot` 类 - 自动驾驶系统

```python
class Autopilot:
    """
    自动驾驶系统
    
    实现标准民航飞行程序，包括：
    - SID (标准离场程序)
    - 巡航导航
    - STAR (标准到达程序)
    - 五边进近
    """
    
    def __init__(
        self,
        dm: DynamicsModel,
        *,
        mode: Literal["standard", "fuel_optimal", "time_optimal"] = "standard",
        cruise_speed_mach: Optional[float] = None,
        cruise_altitude_ft: Optional[float] = None
    ):
        """
        初始化自动驾驶
        
        参数:
            dm: 动力学模型实例
            mode: 飞行模式
            cruise_speed_mach: 巡航马赫数
            cruise_altitude_ft: 巡航高度 (英尺)
        """
    
    # ===== 航线加载 =====
    
    def load_route(
        self,
        route: Union["Route", List[Tuple[float, float]]],
        departure_runway_heading: float = 0.0,
        arrival_runway_heading: Optional[float] = None
    ) -> None:
        """加载航线"""
    
    # ===== 控制更新 =====
    
    def update(self) -> Tuple[float, float, float]:
        """
        计算当前仿真步的控制输入
        
        返回:
            (throttle, pitch_target, roll_target)
        """
    
    @property
    def phase(self) -> "FlightPhase":
        """当前飞行阶段"""
    
    @property
    def is_complete(self) -> bool:
        """飞行是否完成"""
    
    @property
    def route_progress(self) -> float:
        """航线进度 (0.0 - 1.0)"""
    
    @property
    def distance_to_destination(self) -> float:
        """到目的地距离 (米)"""


class FlightPhase(Enum):
    """飞行阶段枚举"""
    TAXI = "TAXI"                       # 滑行
    TAKEOFF_ROLL = "TAKEOFF_ROLL"       # 起飞滑跑
    ROTATION = "ROTATION"               # 抬轮
    INITIAL_CLIMB = "INITIAL_CLIMB"     # 初始爬升
    DEPARTURE = "DEPARTURE"             # 离场
    CLIMB = "CLIMB"                     # 爬升
    CRUISE = "CRUISE"                   # 巡航
    DESCENT = "DESCENT"                 # 下降
    APPROACH_DESCENT = "APPROACH_DESCENT"  # 进近下降
    DOWNWIND = "DOWNWIND"               # 三边（下风边）
    BASE = "BASE"                       # 四边（基线边）
    FINAL = "FINAL"                     # 五边（最后进近）
    FLARE = "FLARE"                     # 拉平着陆
    TOUCHDOWN = "TOUCHDOWN"             # 接地
    ROLLOUT = "ROLLOUT"                 # 着陆滑跑
    GO_AROUND = "GO_AROUND"             # 复飞
```

### 5.3 `AircraftParams` 类 - 飞机参数

```python
@dataclass
class AircraftParams:
    """
    飞机完整性能参数（只读数据类）
    
    所有数据来源于 data/plans.csv
    """
    
    # 基本信息
    aircraft_type: str          # 型号
    manufacturer: str           # 制造商
    
    # 质量参数
    mtow_kg: float              # 最大起飞重量 (kg)
    oew_kg: float               # 空机运营重量 (kg)
    max_payload_kg: float       # 最大有效载荷 (kg)
    fuel_capacity_kg: float     # 燃油容量 (kg)
    
    # 几何参数
    wing_area_m2: float         # 机翼面积 (m²)
    wingspan_m: float           # 翼展 (m)
    length_m: float             # 机身长度 (m)
    height_m: float             # 机身高度 (m)
    
    # 动力参数
    max_thrust_n: float         # 单发最大推力 (N)
    num_engines: int            # 发动机数量
    
    # 性能参数
    cruise_speed_mach: float    # 典型巡航马赫数
    cruise_speed_mach_min: float  # 最小巡航马赫数
    cruise_speed_mach_max: float  # 最大巡航马赫数
    cruise_alt_ft: float        # 典型巡航高度 (ft)
    max_range_km: float         # 最大航程 (km)
    service_ceiling_ft: float   # 实用升限 (ft)
    
    # 气动参数
    cl_alpha: float             # 升力线斜率
    cl_max_clean: float         # 光洁构型最大升力系数
    cl_max_takeoff: float       # 起飞构型最大升力系数
    cl_max_landing: float       # 着陆构型最大升力系数
    cd_0_clean: float           # 光洁构型零升阻力系数
    cd_0_takeoff: float         # 起飞构型零升阻力系数
    cd_0_landing: float         # 着陆构型零升阻力系数
    k_induced: float            # 诱导阻力因子
    gear_drag_cd: float         # 起落架阻力增量
    
    # 操纵限制
    max_roll_rate_deg_s: float  # 最大滚转速率 (度/秒)
    max_pitch_rate_deg_s: float # 最大俯仰速率 (度/秒)
    
    # 着陆参数
    approach_speed_kts: float   # 进近速度 (节)
    landing_distance_m: float   # 着陆距离 (米)
    takeoff_distance_m: float   # 起飞距离 (米)
    
    # 运营信息
    typical_pax: int            # 典型座位数
    range_category: str         # 航程类别
    
    # ===== 计算属性 =====
    
    @property
    def typical_mass_kg(self) -> float:
        """典型任务质量"""
    
    @property
    def aspect_ratio(self) -> float:
        """展弦比"""
    
    @property
    def wing_loading(self) -> float:
        """翼载荷 (kg/m²)"""
    
    @property
    def thrust_to_weight(self) -> float:
        """推重比"""
    
    @property
    def total_max_thrust(self) -> float:
        """全部发动机总推力 (N)"""
```

### 5.4 `NavUtils` 类 - 导航工具

```python
class NavUtils:
    """
    导航计算工具类（全静态方法）
    """
    
    R_EARTH = 6371000.0  # 地球半径 (米)
    
    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        计算两点间的大圆距离 (Haversine 公式)
        
        返回: 距离 (米)
        """
    
    @staticmethod
    def calculate_bearing(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        计算从起点到终点的航向角
        
        返回: 航向 (度, 0-360, 0=北)
        """
    
    @staticmethod
    def intermediate_point(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        fraction: float
    ) -> Tuple[float, float]:
        """
        计算大圆航线上的中间点
        
        参数:
            fraction: 比例 (0=起点, 1=终点)
        
        返回: (lat, lon)
        """
    
    @staticmethod
    def destination_point(
        lat: float, lon: float,
        bearing: float, distance: float
    ) -> Tuple[float, float]:
        """
        给定起点、航向和距离，计算目的点
        
        返回: (lat, lon)
        """
    
    @staticmethod
    def cross_track_distance(
        lat: float, lon: float,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        计算点到大圆航线的垂直距离
        
        返回: 距离 (米)，正值表示点在航线右侧
        """
    
    @staticmethod
    def along_track_distance(
        lat: float, lon: float,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        计算点沿航线方向到起点的距离
        
        返回: 距离 (米)
        """
```

---

## 六、模块结构

```
flightsim/
├── __init__.py              # 统一导出入口
│
├── api.py                   # 高层 API
│   ├── generate_trajectory()
│   ├── list_aircraft()
│   ├── list_routes()
│   └── get_airport()
│
├── aircraft.py              # 中层 API - 飞机类
│   └── Aircraft
│
├── trajectory.py            # 中层 API - 轨迹类
│   └── Trajectory
│
├── route.py                 # 中层 API - 航线类
│   └── Route
│
├── dynamics.py              # 底层 API - 动力学模型
│   └── DynamicsModel
│
├── autopilot.py             # 底层 API - 自动驾驶
│   ├── Autopilot
│   └── FlightPhase
│
├── aerodynamics.py          # 底层 API - 飞机参数
│   ├── AircraftParams
│   └── AircraftDatabase
│
├── navigation.py            # 工具 - 导航计算
│   └── NavUtils
│
├── noise.py                 # 工具 - 噪声模型
│   └── NoiseGenerator
│
├── data/                    # 数据文件
│   ├── plans.csv            # 飞机参数
│   ├── waypoints.csv        # 预设航线
│   └── airports.csv         # 机场数据 (新增)
│
└── exceptions.py            # 自定义异常
    ├── AircraftNotFoundError
    ├── RouteNotFoundError
    └── SimulationError
```

---

## 七、`__init__.py` 导出设计

```python
"""
FlightSim - 快速飞行轨迹仿真库

基于六自由度动力学模型的民航飞行轨迹生成器。

快速开始:
    >>> import flightsim as fs
    >>> 
    >>> # 一行代码生成航迹
    >>> df = fs.generate_trajectory("B737-800", "PEK", "PVG")
    >>> 
    >>> # 查看可用机型
    >>> fs.list_aircraft()
    >>> 
    >>> # 高级用法
    >>> plane = fs.Aircraft("A320-200")
    >>> traj = plane.fly("PEK", "CAN", cruise_altitude_ft=35000)
"""

__version__ = "1.0.0"

# ========== 高层 API (推荐) ==========
from .api import (
    generate_trajectory,
    list_aircraft,
    list_routes,
    get_airport,
)

# ========== 中层 API ==========
from .aircraft import Aircraft
from .trajectory import Trajectory
from .route import Route

# ========== 底层 API ==========
from .dynamics import DynamicsModel
from .autopilot import Autopilot, FlightPhase
from .aerodynamics import AircraftParams, AircraftDatabase

# ========== 工具 ==========
from .navigation import NavUtils

# ========== 异常 ==========
from .exceptions import (
    AircraftNotFoundError,
    RouteNotFoundError,
    SimulationError,
)

__all__ = [
    # 版本
    "__version__",
    # 高层 API
    "generate_trajectory",
    "list_aircraft", 
    "list_routes",
    "get_airport",
    # 中层 API
    "Aircraft",
    "Trajectory",
    "Route",
    # 底层 API
    "DynamicsModel",
    "Autopilot",
    "FlightPhase",
    "AircraftParams",
    "AircraftDatabase",
    # 工具
    "NavUtils",
    # 异常
    "AircraftNotFoundError",
    "RouteNotFoundError",
    "SimulationError",
]
```

---

## 八、使用示例汇总

### 场景 1: 快速生成航迹（新手）

```python
import flightsim as fs

# 最简调用
df = fs.generate_trajectory("B737-800", "PEK", "PVG")
print(df.head())

# 保存到文件
fs.generate_trajectory(
    "A320-200", 
    "PEK", "CAN",
    output_file="flight_pek_can.csv"
)
```

### 场景 2: 定制飞行参数（中级用户）

```python
import flightsim as fs

# 创建飞机
plane = fs.Aircraft("B777-300ER")

# 自定义飞行
traj = plane.fly(
    origin="PEK",
    destination="LAX",
    cruise_altitude_ft=41000,
    cruise_speed_mach=0.85,
    time_step=1.0
)

# 分析结果
print(traj.summary)
traj.plot_profile()
traj.to_csv("pek_lax_flight.csv")
```

### 场景 3: 手动控制动力学模型（高级用户）

```python
import flightsim as fs

# 创建动力学模型
dm = fs.DynamicsModel(
    "A320-200",
    start_lat=40.08,
    start_lon=116.58,
    start_alt=10.0,
    start_heading=90.0,
    dt=0.5
)

# 起飞阶段
dm.set_config(flaps=1, gear_down=True)
states = []

# 起飞滑跑
for _ in range(200):
    state = dm.step(throttle=1.0, pitch_target=0, roll_target=0)
    states.append(state)
    if state['tas'] > 70:  # 达到抬轮速度
        break

# 抬轮爬升
for _ in range(100):
    state = dm.step(throttle=0.95, pitch_target=15, roll_target=0)
    states.append(state)

# 收起落架和襟翼
dm.set_config(flaps=0, gear_down=False)

# 继续爬升...
print(f"当前高度: {dm.altitude:.0f}m, 速度: {dm.speed:.1f}m/s")
```

### 场景 4: 使用自动驾驶

```python
import flightsim as fs

# 创建模型和自动驾驶
dm = fs.DynamicsModel("B737-800", start_lat=40.08, start_lon=116.58, dt=0.5)
ap = fs.Autopilot(dm, cruise_altitude_ft=35000)

# 加载航线
route = fs.Route.from_airports("PEK", "SHA")
ap.load_route(route)

# 仿真循环
while not ap.is_complete:
    throttle, pitch, roll = ap.update()
    state = dm.step(throttle, pitch, roll)
    
    if dm.time % 60 == 0:
        print(f"时间: {dm.time/60:.0f}min, 阶段: {ap.phase.value}, 高度: {state['alt']:.0f}m")
```

---

## 九、参数默认值策略

| 参数 | 未指定时的默认行为 |
|------|-------------------|
| **航线** | 1. 查找预设航线 → 2. 生成大圆直飞 |
| **巡航高度** | 使用 `plans.csv` 中机型的 `cruise_alt_ft` |
| **巡航速度** | 使用 `plans.csv` 中机型的 `cruise_speed_mach` |
| **初始燃油** | 满油 (`fuel_capacity_kg`) |
| **载荷** | 典型载荷 (`typical_pax * 100kg`) |
| **时间步长** | 0.5 秒 |
| **跑道方向** | 自动计算（起点到第一个航路点的方向）|

---

## 十、验证计划

由于这是一个 API 设计方案文档，不涉及代码实现，验证将在用户审阅并确认方案后进行。

### 后续实现阶段的验证方式

1. **单元测试**: 对每个类和函数编写测试用例
2. **集成测试**: 测试完整的飞行仿真流程
3. **示例验证**: 确保所有文档中的示例代码可运行
4. **对比验证**: 与现有 `examples/simple_trajectory.py` 的输出对比

---

> [!IMPORTANT]
> 此方案为设计文档，需要用户审阅确认后再进行代码实现。
