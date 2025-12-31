"""
生成带有详细飞行阶段的轨迹数据
展示完整的飞行过程，包括五边进近
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flightsim.sixdof import SixDOFModel
from flightsim.autopilot import StandardAutopilot, FlightPhase
from flightsim.navigation import NavUtils


def generate_flight_trajectory(
    aircraft_type: str,
    origin: tuple,
    destination: tuple,
    waypoints: list = None,
    output_file: str = None,
    dt: float = 0.5,
    max_time: float = 14400  # 最大飞行时间4小时
):
    """
    生成完整的飞行轨迹
    
    Args:
        aircraft_type: 飞机型号
        origin: 起点 (lat, lon)
        destination: 终点 (lat, lon)
        waypoints: 中间航路点列表
        output_file: 输出CSV文件名
        dt: 时间步长（秒）
        max_time: 最大仿真时间（秒）
        
    Returns:
        DataFrame: 轨迹数据
    """
    print(f"\n{'='*80}")
    print(f"Flight Trajectory Generation")
    print(f"{'='*80}")
    print(f"Aircraft: {aircraft_type}")
    print(f"Origin: {origin}")
    print(f"Destination: {destination}")
    
    # 构建完整航路
    if waypoints is None:
        waypoints = []
    
    full_route = [origin] + waypoints + [destination]
    
    # 计算航线总距离
    total_distance = 0
    for i in range(len(full_route) - 1):
        dist = NavUtils.haversine_distance(
            full_route[i][0], full_route[i][1],
            full_route[i+1][0], full_route[i+1][1]
        )
        total_distance += dist
    
    print(f"Total route distance: {total_distance/1000:.1f} km")
    print(f"Waypoints: {len(full_route)}")
    
    # 计算跑道方向（简化：使用起点到第一个航路点的方向）
    if len(full_route) > 1:
        runway_heading = NavUtils.calculate_bearing(
            origin[0], origin[1], full_route[1][0], full_route[1][1]
        )
    else:
        runway_heading = 0
    
    # 初始化模型
    model = SixDOFModel(
        aircraft_type=aircraft_type,
        start_lat=origin[0],
        start_lon=origin[1],
        start_alt=10.0,  # 机场高程
        start_heading=runway_heading,
        dt=dt
    )
    
    # 初始化自动驾驶
    autopilot = StandardAutopilot(model)
    autopilot.load_route(full_route, departure_alt=10.0, runway_heading=runway_heading)
    
    # 仿真循环
    trajectory = []
    time = 0
    
    print(f"\nStarting simulation...")
    print(f"{'Time':>8s} {'Phase':>20s} {'Alt':>8s} {'Speed':>8s} {'Dist':>10s}")
    print(f"{'-'*80}")
    
    last_print_time = 0
    
    while time < max_time:
        # 获取控制指令
        throttle, pitch, roll = autopilot.update()
        
        # 更新动力学
        model.update(throttle, pitch, roll)
        
        # 获取状态
        state = model.get_state()
        phase = autopilot.get_phase()
        
        # 计算到目的地距离
        dist_to_dest = NavUtils.haversine_distance(
            state['lat'], state['lon'],
            destination[0], destination[1]
        )
        
        # 记录轨迹
        trajectory.append({
            'time': time,
            'lat': state['lat'],
            'lon': state['lon'],
            'alt': state['alt'],
            'tas': state['tas'],
            'ias': state['ias'],
            'gs': state['gs'],
            'mach': state['mach'],
            'heading': state['heading'],
            'pitch': state['pitch'],
            'roll': state['roll'],
            'gamma': state['gamma'],
            'alpha': state['alpha'],
            'v_vertical': state['speed_v'],
            'throttle': throttle,
            'lift': state['lift'],
            'drag': state['drag'],
            'thrust': state['thrust'],
            'mass': state['mass'],
            'fuel': state['fuel'],
            'load_factor': state['load_factor'],
            'flaps': state['flaps'],
            'gear': state['gear'],
            'spoilers': state['spoilers'],
            'flight_phase': phase.value,  # 飞行阶段
            'dist_to_dest': dist_to_dest
        })
        
        # 打印进度
        if time - last_print_time >= 60 or phase != autopilot.phase:
            print(f"{time:8.0f} {phase.value:>20s} {state['alt']:8.0f} "
                  f"{state['tas']:8.1f} {dist_to_dest/1000:10.1f}")
            last_print_time = time
        
        # 终止条件
        if phase == FlightPhase.ROLLOUT and state['tas'] < 10:
            print(f"\nFlight completed at t={time:.0f}s")
            break
        
        time += dt
    
    # 转换为DataFrame
    df = pd.DataFrame(trajectory)
    
    # 保存到文件
    if output_file:
        output_path = Path(__file__).parent / output_file
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\nTrajectory saved to: {output_path}")
        print(f"Total points: {len(df)}")
        print(f"Duration: {time/60:.1f} minutes")
    
    # 统计信息
    print(f"\n{'='*80}")
    print(f"Flight Statistics")
    print(f"{'='*80}")
    print(f"Total time: {time/60:.1f} minutes")
    print(f"Total distance: {total_distance/1000:.1f} km")
    print(f"Average speed: {total_distance/time*3.6:.1f} km/h")
    print(f"Max altitude: {df['alt'].max():.0f} m ({df['alt'].max()*3.28084:.0f} ft)")
    print(f"Max speed: {df['tas'].max():.1f} m/s ({df['tas'].max()*1.944:.0f} kts)")
    print(f"Fuel used: {df['fuel'].iloc[0] - df['fuel'].iloc[-1]:.0f} kg")
    
    # 飞行阶段统计
    print(f"\nFlight Phase Duration:")
    phase_times = df.groupby('flight_phase')['time'].agg(['count', 'min', 'max'])
    phase_times['duration_min'] = (phase_times['max'] - phase_times['min']) / 60
    for phase, row in phase_times.iterrows():
        print(f"  {phase:20s}: {row['duration_min']:6.1f} min")
    
    print(f"{'='*80}\n")
    
    return df


def main():
    """主函数：生成示例轨迹"""
    
    # 示例1：短程航线（北京-上海）
    print("\n" + "="*80)
    print("Example 1: Short-haul Flight (PEK-PVG)")
    print("="*80)
    
    df1 = generate_flight_trajectory(
        aircraft_type="A320-200",
        origin=(40.0799, 116.6031),  # 北京首都机场
        destination=(31.1443, 121.8083),  # 上海浦东机场
        waypoints=[
            (38.5, 118.0),  # 中间航路点
        ],
        output_file="trajectory_pek_pvg_a320.csv",
        dt=0.5
    )
    
    # 示例2：中程航线（北京-广州）
    print("\n" + "="*80)
    print("Example 2: Medium-haul Flight (PEK-CAN)")
    print("="*80)
    
    df2 = generate_flight_trajectory(
        aircraft_type="B737-800",
        origin=(40.0799, 116.6031),  # 北京
        destination=(23.3924, 113.2988),  # 广州
        waypoints=[
            (36.0, 115.0),
            (30.0, 114.0),
        ],
        output_file="trajectory_pek_can_b737.csv",
        dt=0.5
    )
    
    # 示例3：长程航线（北京-洛杉矶）- 使用真实航线数据
    print("\n" + "="*80)
    print("Example 3: Long-haul Flight (PEK-LAX)")
    print("="*80)
    
    # 从 data/waypoints.csv读取真实航路点
    waypoints_df = pd.read_csv(PROJECT_ROOT / "data" / "waypoints.csv")
    pek_lax = waypoints_df[
        (waypoints_df['origin_code'] == 'PEK') & 
        (waypoints_df['dest_code'] == 'LAX')
    ].iloc[0]
    
    # 提取航路点
    route_waypoints = []
    for i in range(1, 11):
        lat = pek_lax.get(f'waypoint{i}_lat')
        lon = pek_lax.get(f'waypoint{i}_lon')
        if pd.notna(lat) and pd.notna(lon):
            route_waypoints.append((lat, lon))
    
    df3 = generate_flight_trajectory(
        aircraft_type="B777-300ER",
        origin=(pek_lax['origin_lat'], pek_lax['origin_lon']),
        destination=(pek_lax['dest_lat'], pek_lax['dest_lon']),
        waypoints=route_waypoints,
        output_file="trajectory_pek_lax_b777.csv",
        dt=5.0  # 长程航线使用更大时间步长
    )
    
    print("\n" + "="*80)
    print("All trajectories generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

