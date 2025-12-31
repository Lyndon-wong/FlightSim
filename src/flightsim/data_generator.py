"""
数据生成工具
生成带噪声的轨迹数据，用于对比和测试
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from .navigation import NavUtils


def generate_mock_real_data(waypoints: List[Tuple[float, float]], noise_level: float = 1.0, total_points: int = 1000) -> pd.DataFrame:
    """
    基于大圆航线生成带噪声的ADS-B数据
    
    Args:
        waypoints: 航路点列表，每个点为(lat, lon)元组
        noise_level: 噪声水平（1.0为正常水平）
        total_points: 总点数
        
    Returns:
        包含lat, lon, alt, source列的DataFrame
    """
    lat_sim = []
    lon_sim = []
    alt_sim = []
    
    full_path_lats = []
    full_path_lons = []
    full_path_alts = []
    
    # 构建理想的大圆骨架
    for i in range(len(waypoints)-1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        
        # 每一段分配的点数（根据距离分配更合理，这里简化为均分）
        segment_points = int(total_points / (len(waypoints)-1))
        
        for j in range(segment_points):
            frac = j / segment_points
            lat, lon = NavUtils.intermediate_point(p1[0], p1[1], p2[0], p2[1], frac)
            full_path_lats.append(lat)
            full_path_lons.append(lon)
            
            # 高度剖面（简化）
            global_progress = (i * segment_points + j) / total_points
            if global_progress < 0.1:
                h = global_progress * 10 * 11000  # Climb
            elif global_progress > 0.9:
                h = (1.0 - global_progress) * 10 * 11000  # Descent
            else:
                h = 11000  # Cruise
            full_path_alts.append(h)

    # 添加噪声
    # 风偏通常是低频的大波动（Red Noise），传感器是高频小波动（White Noise）
    wind_lat_err = 0
    wind_lon_err = 0
    
    for k in range(len(full_path_lats)):
        # 随机游走模拟风偏
        wind_lat_err += np.random.normal(0, 0.005 * noise_level)
        wind_lon_err += np.random.normal(0, 0.005 * noise_level)
        # 限制最大风偏
        wind_lat_err = np.clip(wind_lat_err, -0.2, 0.2)
        wind_lon_err = np.clip(wind_lon_err, -0.2, 0.2)
        
        # 传感器噪声
        sensor_lat = np.random.normal(0, 0.001 * noise_level)
        sensor_lon = np.random.normal(0, 0.001 * noise_level)
        
        lat_sim.append(full_path_lats[k] + wind_lat_err + sensor_lat)
        lon_sim.append(full_path_lons[k] + wind_lon_err + sensor_lon)
        alt_sim.append(full_path_alts[k] + np.random.normal(0, 15))
    
    return pd.DataFrame({
        'lat': lat_sim,
        'lon': lon_sim,
        'alt': alt_sim,
        'source': 'Real Data (Mock)'
    })

