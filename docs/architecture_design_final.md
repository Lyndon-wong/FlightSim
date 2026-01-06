# FlightSim 重构设计方案 V4 (Final Detailed Proposal)

本方案结合了 V2 的领域深度与 V3 的扩展性设计，提供最详尽的架构规划。核心目标是将 `FlightSim` 打造为一个**高保真、模块化、可扩展**的飞行仿真科研平台。

## 1. 总体架构理念 (Architectural Philosophy)

系统被严格划分为三个物理层级和一套交互接口：

1.  **Environment (环境层)**: 提供仿真所需的物理世界边界（大气、重力、磁场、地形）。
2.  **Plane (对象层)**: 被控实体。包含物理动力学真值内核、传感器模型（带噪声）和执行机构模型。
3.  **GNC (智能层)**: 飞行控制软件。包含导航滤波、制导律、控制律。
4.  **Interfaces (接口层)**: 定义上述模块交互的标准协议，支持第三方算法注入。

---

## 2. 详细目录结构与文件职责 (Detailed Structure & Responsibilities)

以下结构展示了完整的源码组织及其功能定义。

```text
src/flightsim/
├── api/                            # [API 层] 对外暴露的统一调用接口
│   ├── __init__.py
│   └── simulation.py               # FlightSim类。负责组装各模块，运行主循环 (step)，管理仿真时钟。
│
├── interfaces.py                   # [接口定义] 核心抽象基类 (ABC)。定义 Navigator, ControlLaw, AircraftDesign 等标准。
│
├── environment/                    # [环境层] 物理世界的静态与动态属性
│   ├── __init__.py
│   ├── atmosphere.py               # ISA标准大气模型。计算 rho, temp, pressure, speed_of_sound。
│   ├── gravity.py                  # 重力模型 (支持 WGS84 椭球重力)。
│   ├── mag_field.py                # 地磁模型 (WMM)。用于磁力计仿真。
│   ├── wind.py                     # 风场模型。定义平均风 (Mean Wind)。
│   ├── turbulence.py               # 湍流/阵风模型 (Dryden/Von Karman)。为动力学和气动提供扰动输入。
│   └── world.py                    # 静态世界数据。包含机场数据库 (Airports)、地形高程 (Terrain)。
│
├── plane/                          # [对象层 -> 飞机] 物理实体
│   ├── __init__.py
│   ├── aircraft.py                 # 飞机顶层容器。组合 Dynamics, Sensors, Actuators 组件。
│   │
│   ├── dynamics/                   # [动力学内核] 计算物理真值 (Ground Truth)
│   │   ├── __init__.py
│   │   ├── core.py                 # 6-DOF 运动方程 (EOM) 求解器。积分位置、速度、姿态。
│   │   ├── aerodynamics.py         # 气动解算器。计算 CL, CD, SideForce, Moments (含马赫/雷诺修正)。
│   │   ├── propulsion.py           # 推进系统。计算推力 (Thrust) 和 燃油流率 (FF)。
│   │   └── mass_properties.py      # 质量特性。计算实时质量、惯性张量 (Inertia Tensor)、重心 (CG) 变化。
│   │
│   ├── sensors/                    # [传感器模型] 叠加误差与噪声
│   │   ├── __init__.py
│   │   ├── imu.py                  # 惯性测量单元。加速度计 (Accel) + 陀螺仪 (Gyro)。包含 Bias, Walk, Whitenoise。
│   │   ├── gnss.py                 # 全球导航卫星系统。位置/速度读数。包含延迟、漂移、多径效应。
│   │   ├── air_data.py             # 大气数据计算机。空速管 (Pitot) -> IAS/TAS, 气压计 -> Baro Alt。
│   │   └── magnetometer.py         # 磁力计。输出磁航向。
│   │
│   └── actuators/                  # [执行机构] 物理动作的响应特性
│       ├── __init__.py
│       └── surfaces.py             # 舵面动力学。模拟舵机响应延迟 (一阶/二阶系统)、速率限制 (Rate Limits)。
│
├── gnc/                            # [智能层 -> GNC] 制导、导航与控制算法
│   ├── __init__.py
│   │
│   ├── navigation/                 # [导航子系统] "我们在哪里？"
│   │   ├── __init__.py
│   │   ├── estimator.py            # 状态估计器基类。
│   │   ├── ekf.py                  # (可选) 扩展卡尔曼滤波实现。融合 IMU + GNSS。
│   │   └── initialization.py       # 导航对准逻辑 (Alignment)。
│   │
│   ├── guidance/                   # [制导子系统] "我们要去哪里？"
│   │   ├── __init__.py
│   │   ├── flight_plan.py          # 飞行计划管理。航路点序列、航段切换逻辑。
│   │   ├── lnav.py                 # 横向制导 (LNAV)。计算目标航迹角、偏航距 (XTE)、切向角。
│   │   ├── vnav.py                 # 垂向制导 (VNAV)。计算目标高度、爬升/下降率剖面 (TOD计算)。
│   │   └── director.py             # 飞行指引 (Flight Director)。综合 LNAV/VNAV 输出目标指令 (Target Alt/Spd/Hdg)。
│   │
│   └── control/                    # [控制子系统] "如何执行？"
│       ├── __init__.py
│       ├── autopilot.py            # 自动驾驶主逻辑。根据当前 Mode 分发控制任务。
│       ├── laws.py                 # 基础控制律。PID 算法、前馈控制、阻尼逻辑。
│       └── stability.py            # 增稳系统 (SAS)。角速率阻尼、包线保护 (Alpha Protection)。
│
├── configs/                        # [配置数据]
│   ├── aircraft/                   # 机型参数定义 (JSON/YAML)
│   │   ├── b737.py                 # 波音737 气动/质量/推力参数类
│   │   └── a320.py                 # 空客A320 参数类
│   └── sensors/                    # 传感器噪声配置
│       └── consumer_grade.py       # 消费级 IMU 参数
│
└── utils/                          # [通用工具]
    ├── math_utils.py               # 坐标转换 (LLA <-> ECEF <-> NED)、四元数运算。
    ├── units.py                    # 单位转换常量。
    └── logging.py                  # 仿真数据记录工具 (CSV/Binary)。
```

## 3. 核心扩展接口设计 (Core Extensibility Interfaces)

为了支持“人员将自己的方法提交进来”，以下接口是关键扩展点：

### 3.1 导航接口 (`Navigator`)
用户可替换默认的导航逻辑（如引入自己的视觉导航或融合算法）。
```python
class Navigator(ABC):
    @abstractmethod
    def update(self, sensors: SensorData, dt: float) -> EstimatedState:
        """接收传感器读数，输出估计状态"""
        pass
```

### 3.2 控制律接口 (`ControlLaw`)
用户可验证先进控制算法（如 LQR, MPC, 自适应控制）。
```python
class ControlLaw(ABC):
    @abstractmethod
    def calculate_controls(self, 
                           target: GuidanceCommand, 
                           current: EstimatedState, 
                           dt: float) -> ActuatorCommands:
        """接收制导指令和当前状态，输出舵面/油门指令"""
        pass
```

### 3.3 飞机设计接口 (`AircraftDesign`)
用户可注入全新的飞行器物理模型（如 eVTOL, 无人机）。
```python
class AircraftDesign(ABC):
    def get_aero_coeffs(self, alpha, beta, mach, rates, controls) -> AeroForces:
        """计算气动力与力矩"""
        pass
    def get_thrust(self, throttle, alt, mach) -> ThrustForces:
        """计算推力"""
        pass
```

## 4. API 调用示例
对外提供的标准调用方式，隐藏内部复杂性：

```python
from flightsim.api import FlightSim
from flightsim.configs.aircraft import B737_800
from myspace.my_research import SuperEKF, RobustController # 用户自定义模块

# 1. 初始化仿真，注入自定义算法
sim = FlightSim(
    aircraft=B737_800(),
    navigator=SuperEKF(),       # 注入自定义导航
    controller=RobustController() # 注入自定义控制
)

# 2. 设置任务
sim.load_route([(30, 120), (31, 121)])

# 3. 运行循环
for t in range(1000):
    state = sim.step(dt=0.1)
    print(f"Time: {t}, Alt: {state.estimated.alt}, Mode: {state.autopilot_mode}")
```
