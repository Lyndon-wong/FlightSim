"""
验证控制律鲁棒性脚本
测试在强侧风下的航迹跟踪能力
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flightsim.sixdof import SixDOFModel
from flightsim.autopilot import StandardAutopilot, FlightPhase
from flightsim.noise import NoiseConfig
from flightsim.navigation import NavUtils

def run_robustness_test():
    print(f"\n{'='*80}")
    print(f"Robustness Verification Test")
    print(f"{'='*80}")
    
    # 1. 设置强侧风场景
    # 模拟湍流
    noise_config = NoiseConfig(
        wind_intensity=0.5   # 中度湍流
    )
    
    # 起点和终点 (正北航向)
    origin = (30.0, 120.0)
    destination = (32.0, 120.0) # 正北 2度纬度 (~220km)
    
    print(f"Scenario: Flying North (360 deg) with 20 m/s STEADY Crosswind")
    print(f"Expected: Aircraft should crab to maintain North track")
    
    # 初始化模型
    model = SixDOFModel(
        aircraft_type="A320-200",
        start_lat=origin[0],
        start_lon=origin[1],
        start_alt=2000.0,  # 空中开始
        start_heading=0.0,
        dt=0.1,  # 使用更小的时间步长以获得更高精度
        noise_config=noise_config
    )
    
    # 初始化自动驾驶
    autopilot = StandardAutopilot(model)
    # 只需要两个点：起点和终点
    autopilot.load_route([origin, destination], departure_alt=0.0, runway_heading=0.0)
    
    # 手动设置阶段为巡航，跳过起飞
    autopilot.phase = FlightPhase.CRUISE
    autopilot.cruise_alt_m = 2000.0
    autopilot.target_alt = 2000.0
    model.tas = 230.0 # 初始速度
    model.set_config(flaps_idx=0, gear_down=False)
    
    # 手动设置上一航路点以激活XTE
    autopilot.prev_wp = origin
    
    # 数据记录
    history = []
    max_time = 600.0 # 测试10分钟
    time = 0.0
    
    print(f"\nStarting simulation...")
    
    while time < max_time:
        # 强制注入稳态侧风 (20 m/s)
        # 这是一个Hack，因为默认Dryden模型是零均值的湍流
        model.gust_v = 20.0 
        
        # 更新
        throttle, pitch, roll = autopilot.update()
        model.update(throttle, pitch, roll)
        
        state = model.get_state()
        
        # 计算偏航距离 (XTE)
        # 航线是正北 (经度 120.0)
        # 简单计算：经度偏差转换为距离
        # lat 1 deg ~ 111km
        # lon 1 deg ~ 111 * cos(lat) km
        d_lon = state['lon'] - 120.0
        xte_m = d_lon * 111320 * np.cos(np.radians(state['lat']))
        
        history.append({
            'time': time,
            'xte': xte_m,
            'alt_error': state['alt'] - 2000.0,
            'heading': state['heading'],
            'roll': state['roll'],
            'wind_v': state['gust_v']
        })
        
        if time % 60 < 0.1:
            print(f"T={time:5.1f}s | XTE={xte_m:6.1f}m | AltErr={state['alt']-2000.0:5.1f}m | Hdg={state['heading']:5.1f} | Roll={state['roll']:5.1f}")
            
        time += 0.1
        
    # 分析结果
    df = pd.DataFrame(history)
    
    # 去掉前60秒的过渡期
    stable_df = df[df['time'] > 60]
    
    print(f"\n{'='*80}")
    print(f"Test Results (Stable Phase > 60s)")
    print(f"{'='*80}")
    
    max_xte = stable_df['xte'].abs().max()
    mean_xte = stable_df['xte'].abs().mean()
    mean_alt_err = stable_df['alt_error'].mean()
    
    print(f"Max Cross-Track Error: {max_xte:.2f} m")
    print(f"Mean Cross-Track Error: {mean_xte:.2f} m")
    print(f"Mean Altitude Error: {mean_alt_err:.2f} m")
    
    # 判定标准
    success = True
    if max_xte > 200.0: # 允许最大200米偏差
        print(f"[FAIL] XTE too large (> 200m)")
        success = False
    else:
        print(f"[PASS] XTE within limits")
        
    if abs(mean_alt_err) > 5.0: # 允许5米高度误差
        print(f"[FAIL] Altitude error too large (> 5m)")
        success = False
    else:
        print(f"[PASS] Altitude error within limits")
        
    output_file = PROJECT_ROOT / "examples" / "robustness_test_result.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed data saved to {output_file}")
    
    return success

if __name__ == "__main__":
    run_robustness_test()
