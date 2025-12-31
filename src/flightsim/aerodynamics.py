"""
气动模型和飞机参数定义
支持从plans.csv加载真实飞机参数
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class AeroTables:
    """气动数据表"""
    cl_alpha: float  # 升力系数斜率 (per degree)
    cl_max: List[float]  # 最大升力系数 [clean, takeoff, landing]
    alpha_stall: List[float]  # 失速角 [clean, takeoff, landing] (degrees)
    cd_0: List[float]  # 零升阻力系数 [clean, takeoff, landing]
    k: float  # 诱导阻力系数
    gear_drag: float  # 起落架阻力系数


@dataclass
class PerformanceData:
    """性能数据"""
    approach_speed_kts: float  # 进近速度 (节)
    landing_distance_m: float  # 着陆距离 (米)
    takeoff_distance_m: float  # 起飞距离 (米)
    service_ceiling_ft: float  # 实用升限 (英尺)
    max_range_km: float  # 最大航程 (公里)
    cruise_speed_mach: float  # 巡航马赫数
    cruise_alt_ft: float  # 巡航高度 (英尺)


@dataclass
class AircraftParams:
    """完整飞机参数"""
    # 基本信息
    aircraft_type: str  # 飞机型号
    manufacturer: str  # 制造商
    typical_pax: int  # 典型载客量
    
    # 质量参数 (kg)
    mtow_kg: float  # 最大起飞重量
    oew_kg: float  # 使用空重
    max_payload_kg: float  # 最大载荷
    fuel_capacity_kg: float  # 燃油容量
    
    # 几何参数
    wing_area_m2: float  # 机翼面积 (m²)
    wingspan_m: float  # 翼展 (m)
    length_m: float  # 机身长度 (m)
    height_m: float  # 机身高度 (m)
    
    # 推进系统
    max_thrust_n: float  # 最大推力 (N)
    num_engines: int  # 发动机数量
    
    # 气动数据
    aero: AeroTables
    
    # 性能数据
    performance: PerformanceData
    
    # 操纵限制
    max_roll_rate: float  # 最大滚转率 (deg/s)
    max_pitch_rate: float  # 最大俯仰率 (deg/s)
    
    @property
    def typical_mass_kg(self) -> float:
        """典型任务质量（空重 + 50%载荷 + 50%燃油）"""
        return self.oew_kg + 0.5 * self.max_payload_kg + 0.5 * self.fuel_capacity_kg
    
    @property
    def aspect_ratio(self) -> float:
        """展弦比"""
        return (self.wingspan_m ** 2) / self.wing_area_m2
    
    @property
    def wing_loading(self) -> float:
        """翼载荷 (kg/m²)"""
        return self.typical_mass_kg / self.wing_area_m2
    
    @property
    def thrust_to_weight(self) -> float:
        """推重比"""
        return self.max_thrust_n / (self.typical_mass_kg * 9.81)


class AircraftDatabase:
    """飞机数据库管理器"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化数据库
        
        Args:
            csv_path: plans.csv文件路径，如果为None则使用默认路径
        """
        if csv_path is None:
            # 默认路径：src/flightsim/../../data/plans.csv
            current_dir = Path(__file__).parent
            csv_path = current_dir.parent.parent / "data" / "plans.csv"
        
        self.csv_path = Path(csv_path)
        self._aircraft_db: Dict[str, AircraftParams] = {}
        self._load_database()
    
    def _load_database(self):
        """从CSV文件加载飞机数据"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Aircraft database not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        for _, row in df.iterrows():
            # 构建气动数据
            aero = AeroTables(
                cl_alpha=row['cl_alpha'],
                cl_max=[row['cl_max_clean'], row['cl_max_to'], row['cl_max_land']],
                alpha_stall=[15.0, 13.0, 11.0],  # 根据襟翼配置的典型失速角
                cd_0=[row['cd_0_clean'], row['cd_0_to'], row['cd_0_land']],
                k=row['k_induced'],
                gear_drag=row['gear_drag_cd']
            )
            
            # 构建性能数据
            performance = PerformanceData(
                approach_speed_kts=row['approach_speed_kts'],
                landing_distance_m=row['landing_distance_m'],
                takeoff_distance_m=row['takeoff_distance_m'],
                service_ceiling_ft=row['service_ceiling_ft'],
                max_range_km=row['max_range_km'],
                cruise_speed_mach=row['cruise_speed_mach'],
                cruise_alt_ft=row['cruise_alt_ft']
            )
            
            # 构建完整参数
            params = AircraftParams(
                aircraft_type=row['aircraft_type'],
                manufacturer=row['manufacturer'],
                typical_pax=int(row['typical_pax']),
                mtow_kg=row['mtow_kg'],
                oew_kg=row['oew_kg'],
                max_payload_kg=row['max_payload_kg'],
                fuel_capacity_kg=row['fuel_capacity_kg'],
                wing_area_m2=row['wing_area_m2'],
                wingspan_m=row['wingspan_m'],
                length_m=row['length_m'],
                height_m=row['height_m'],
                max_thrust_n=row['max_thrust_n'],
                num_engines=int(row['num_engines']),
                aero=aero,
                performance=performance,
                max_roll_rate=row['max_roll_rate_deg_s'],
                max_pitch_rate=row['max_pitch_rate_deg_s']
            )
            
            self._aircraft_db[row['aircraft_type']] = params
    
    def get_aircraft(self, aircraft_type: str) -> AircraftParams:
        """
        获取飞机参数
        
        Args:
            aircraft_type: 飞机型号（如 "B737-800"）
            
        Returns:
            飞机参数对象
            
        Raises:
            ValueError: 如果飞机型号不存在
        """
        if aircraft_type not in self._aircraft_db:
            available = list(self._aircraft_db.keys())
            raise ValueError(
                f"Aircraft '{aircraft_type}' not found in database.\n"
                f"Available aircraft: {', '.join(available)}"
            )
        return self._aircraft_db[aircraft_type]
    
    def list_aircraft(self) -> List[str]:
        """返回所有可用的飞机型号"""
        return list(self._aircraft_db.keys())
    
    def get_aircraft_by_category(self, category: str) -> List[str]:
        """
        按类别筛选飞机
        
        Args:
            category: 类别 ("narrow_body", "wide_body", "regional")
            
        Returns:
            飞机型号列表
        """
        result = []
        for ac_type, params in self._aircraft_db.items():
            if category == "narrow_body" and 100 <= params.typical_pax <= 250:
                result.append(ac_type)
            elif category == "wide_body" and params.typical_pax > 250:
                result.append(ac_type)
            elif category == "regional" and params.typical_pax < 100:
                result.append(ac_type)
        return result
    
    def print_summary(self):
        """打印数据库摘要"""
        print(f"\n{'='*80}")
        print(f"Aircraft Database Summary")
        print(f"{'='*80}")
        print(f"Total aircraft: {len(self._aircraft_db)}")
        print(f"\nBy manufacturer:")
        manufacturers = {}
        for params in self._aircraft_db.values():
            mfr = params.manufacturer
            manufacturers[mfr] = manufacturers.get(mfr, 0) + 1
        for mfr, count in sorted(manufacturers.items()):
            print(f"  {mfr:20s}: {count:2d} types")
        
        print(f"\nBy category:")
        print(f"  Regional (<100 pax):     {len(self.get_aircraft_by_category('regional'))} types")
        print(f"  Narrow-body (100-250):   {len(self.get_aircraft_by_category('narrow_body'))} types")
        print(f"  Wide-body (>250):        {len(self.get_aircraft_by_category('wide_body'))} types")
        print(f"{'='*80}\n")


# 全局数据库实例（延迟加载）
_global_db: Optional[AircraftDatabase] = None


def get_database() -> AircraftDatabase:
    """获取全局数据库实例"""
    global _global_db
    if _global_db is None:
        _global_db = AircraftDatabase()
    return _global_db


def get_aircraft(aircraft_type: str) -> AircraftParams:
    """
    便捷函数：获取飞机参数
    
    Args:
        aircraft_type: 飞机型号
        
    Returns:
        飞机参数对象
    """
    return get_database().get_aircraft(aircraft_type)


def list_aircraft() -> List[str]:
    """便捷函数：列出所有飞机"""
    return get_database().list_aircraft()


if __name__ == "__main__":
    # 测试代码
    db = AircraftDatabase()
    db.print_summary()
    
    # 显示几个示例
    print("\nExample Aircraft Parameters:\n")
    for ac_type in ["B737-800", "A320-200", "B777-300ER"]:
        params = db.get_aircraft(ac_type)
        print(f"{ac_type}:")
        print(f"  Manufacturer: {params.manufacturer}")
        print(f"  Typical mass: {params.typical_mass_kg/1000:.1f} tons")
        print(f"  Wing area: {params.wing_area_m2:.1f} m²")
        print(f"  Max thrust: {params.max_thrust_n/1000:.0f} kN")
        print(f"  Cruise: Mach {params.performance.cruise_speed_mach} @ {params.performance.cruise_alt_ft/1000:.0f}k ft")
        print(f"  Range: {params.performance.max_range_km:.0f} km")
        print(f"  Thrust/Weight: {params.thrust_to_weight:.2f}")
        print()
