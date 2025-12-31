"""
飞行轨迹模拟器 - Gradio Demo
用户选择起始/目的机场和机型，生成并可视化飞行轨迹
使用完整的六自由度动力学模型
"""
import sys
from pathlib import Path
import tempfile

# 添加动力学模块路径 (从 test/dataset/generators/dynamic 向上5级到项目根目录)
# 添加项目根目录到路径 (假设脚本在 FlightSim/examples)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime

# 导入动力学模块
from flightsim.sixdof import SixDOFModel
from flightsim.autopilot import FlightPhase, create_autopilot
from flightsim.navigation import NavUtils
from flightsim.aerodynamics import get_database

# 数据文件路径
DATA_DIR = PROJECT_ROOT / "data"
WAYPOINTS_FILE = DATA_DIR / "waypoints.csv"
PLANS_FILE = DATA_DIR / "plans.csv"

# 加载数据
routes_df = pd.read_csv(WAYPOINTS_FILE)
aircraft_df = pd.read_csv(PLANS_FILE)

# 准备下拉选项
route_options = [f"{row['route_name']} ({row['origin_code']}-{row['dest_code']}) [{row['distance_km']:.0f}km]" 
                 for _, row in routes_df.iterrows()]
aircraft_options = aircraft_df['aircraft_type'].tolist()


def generate_trajectory(aircraft_type: str, route_data: pd.Series, 
                        cruise_speed_mach: float = None, cruise_alt_ft: float = None, 
                        dt: float = 2.0, max_time: float = None):
    """
    使用六自由度模型生成完整飞行轨迹
    
    Args:
        aircraft_type: 飞机型号
        route_data: 航线数据
        cruise_speed_mach: 巡航马赫数（可选，使用默认最优速度时为None）
        cruise_alt_ft: 巡航高度（英尺）（可选，使用默认高度时为None）
        dt: 时间步长（秒）
        max_time: 最大仿真时间（秒）
    
    Returns:
        DataFrame: 轨迹数据
    """
    # 提取航路点
    waypoints = [(route_data['origin_lat'], route_data['origin_lon'])]
    for i in range(1, 11):
        lat = route_data.get(f'waypoint{i}_lat')
        lon = route_data.get(f'waypoint{i}_lon')
        if pd.notna(lat) and pd.notna(lon):
            waypoints.append((lat, lon))
    waypoints.append((route_data['dest_lat'], route_data['dest_lon']))
    
    # 动态计算最大飞行时间（基于航线距离）
    # 假设平均巡航速度约800-900 km/h，加上起降时间缓冲
    route_distance_km = route_data.get('distance_km', 1000)
    if max_time is None:
        # 估算飞行时间：距离/速度 + 起降缓冲（约30分钟）
        # 使用保守的平均速度750 km/h
        estimated_flight_hours = route_distance_km / 750 + 0.5
        # 转换为秒，并增加50%缓冲以确保完整模拟
        max_time = estimated_flight_hours * 3600 * 1.5
        # 最小2小时，最大20小时
        max_time = max(7200, min(max_time, 72000))
    
    # 计算跑道方向
    runway_heading = NavUtils.calculate_bearing(
        waypoints[0][0], waypoints[0][1], waypoints[1][0], waypoints[1][1]
    )
    
    # 初始化模型
    model = SixDOFModel(aircraft_type, waypoints[0][0], waypoints[0][1],
                        10.0, runway_heading, dt)
    
    # 获取机型航程分类
    ac_info = aircraft_df[aircraft_df['aircraft_type'] == aircraft_type].iloc[0]
    range_category = ac_info.get('range_category', None)
    
    # 使用工厂函数创建合适的自动驾驶
    autopilot = create_autopilot(model, range_category, cruise_speed_mach, cruise_alt_ft)
    autopilot.load_route(waypoints, runway_heading=runway_heading)
    autopilot.phase = FlightPhase.TAXI
    model.set_config(flaps_idx=1, gear_down=True)
    
    # 仿真循环
    trajectory = []
    time = 0
    max_iterations = 50000  # 安全保护
    iteration = 0
    
    while iteration < max_iterations and time < max_time:
        throttle, pitch, roll = autopilot.update()
        model.update(throttle, pitch, roll)
        state = model.get_state()
        phase = autopilot.get_phase()
        
        dist_to_dest = NavUtils.haversine_distance(
            state['lat'], state['lon'], waypoints[-1][0], waypoints[-1][1]
        )
        
        trajectory.append({
            'time': time,
            'lat': state['lat'],
            'lon': state['lon'],
            'alt': state['alt'],
            'tas': state['tas'],
            'heading': state['heading'],
            'pitch': state['pitch'],
            'roll': state['roll'],
            'flight_phase': phase.value,
            'throttle': throttle,
            'target_pitch': pitch,      # 控制指令：目标俯仰
            'target_roll': roll,        # 控制指令：目标滚转
            'dist_to_dest': dist_to_dest,
            'fuel': state['fuel'],
            'mass': state.get('mass', state['fuel'] + 50000)  # 飞机质量用于能量计算
        })
        
        # 终止条件：着陆滑跑结束
        if phase == FlightPhase.ROLLOUT and state['tas'] < 10:
            break
        
        # 终止条件：到达目标机场
        vertical_dist = abs(state['alt'] - 10.0)
        dist_3d = np.sqrt(dist_to_dest**2 + vertical_dist**2)
        if dist_3d < 100 and state['alt'] < 20:
            break
        
        time += dt
        iteration += 1
    
    return pd.DataFrame(trajectory)


def create_map_figure(trajectory_df, route):
    """创建地图可视化 - 白色背景，自动缩放到航线区域"""
    fig = go.Figure()
    
    # 飞行轨迹
    fig.add_trace(go.Scattergeo(
        lon=trajectory_df['lon'],
        lat=trajectory_df['lat'],
        mode='lines+markers',
        line=dict(width=2.5, color='rgba(65, 105, 225, 0.8)'),
        marker=dict(
            size=3,
            color=trajectory_df['alt'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='高度 (m)', x=1.02, thickness=15)
        ),
        name='飞行轨迹',
        hovertemplate='时间: %{customdata[0]:.0f}s<br>高度: %{customdata[1]:.0f}m<br>速度: %{customdata[2]:.1f}m/s<extra></extra>',
        customdata=trajectory_df[['time', 'alt', 'tas']].values
    ))
    
    # 起点和终点
    fig.add_trace(go.Scattergeo(
        lon=[trajectory_df['lon'].iloc[0], trajectory_df['lon'].iloc[-1]],
        lat=[trajectory_df['lat'].iloc[0], trajectory_df['lat'].iloc[-1]],
        mode='markers+text',
        marker=dict(size=12, color=['#27ae60', '#e74c3c'], symbol=['circle', 'square']),
        text=[route['origin_code'], route['dest_code']],
        textposition='top center',
        textfont=dict(size=13, color='#333'),
        name='机场',
        showlegend=False
    ))
    
    # 计算地图边界，自动缩放到航线区域 (60% 占比)
    lat_min, lat_max = trajectory_df['lat'].min(), trajectory_df['lat'].max()
    lon_min, lon_max = trajectory_df['lon'].min(), trajectory_df['lon'].max()
    
    # 计算中心点
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # 计算内容跨度
    lat_span = max(lat_max - lat_min, 1.0)
    lon_span = max(lon_max - lon_min, 1.0)
    
    # 统一缩放基准：取经纬度跨度中较大的一个，并基于它计算视图范围
    # 强制保持 1.8:1 的宽屏比例 (避免南北向航线导致地图变成细长条)
    max_span = max(lat_span, lon_span)
    
    # 高度方向：保留约 45% 的留白 (即内容占 55%)
    view_height = max_span / 0.55
    
    # 宽度方向：强制为高度的 1.8 倍
    view_width = view_height * 1.8
    
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor='rgb(243, 243, 243)',
        showocean=True, oceancolor='rgb(230, 245, 255)',
        coastlinecolor='rgb(180, 180, 180)',
        showlakes=True, lakecolor='rgb(200, 230, 255)',
        showcountries=True, countrycolor='rgb(200, 200, 200)',
        # 设置地图范围
        lataxis=dict(range=[lat_center - view_height/2, lat_center + view_height/2]),
        lonaxis=dict(range=[lon_center - view_width/2, lon_center + view_width/2]),
    )
    
    fig.update_layout(
        title=dict(
            text=f"✈️ 航线: {route['route_name']} ({route['origin_code']} → {route['dest_code']})",
            font=dict(size=16, color='#333')
        ),
        height=600,
        autosize=False,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
        geo=dict(bgcolor='white'),
        legend=dict(
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def create_analysis_figure(trajectory_df):
    """创建分析图表 - 高度、速度、飞行阶段、燃油"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('📈 高度剖面', '🚀 速度曲线', '📊 飞行阶段', '⛽ 燃油消耗'),
        horizontal_spacing=0.12,
        vertical_spacing=0.22
    )
    
    time_min = trajectory_df['time'] / 60
    
    # 1. 高度剖面
    fig.add_trace(
        go.Scatter(
            x=time_min, y=trajectory_df['alt'], 
            fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.2)',
            line=dict(color='#4169E1', width=2), 
            name='高度',
            hovertemplate='时间: %{x:.1f}分钟<br>高度: %{y:.0f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 速度曲线 (转换为节)
    speed_knots = trajectory_df['tas'] * 1.944
    fig.add_trace(
        go.Scatter(
            x=time_min, y=speed_knots,
            line=dict(color='#e74c3c', width=2), 
            name='速度',
            hovertemplate='时间: %{x:.1f}分钟<br>速度: %{y:.0f}节<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. 飞行阶段时间轴 - 修复：按时间顺序排列，正确显示所有阶段
    import plotly.express as px
    phase_colors_list = px.colors.qualitative.Set2
    
    # 获取按首次出现时间排序的阶段列表
    phase_first_time = trajectory_df.groupby('flight_phase')['time'].min().sort_values()
    ordered_phases = phase_first_time.index.tolist()
    
    for i, phase in enumerate(ordered_phases):
        mask = trajectory_df['flight_phase'] == phase
        times = trajectory_df[mask]['time'] / 60
        if len(times) > 0:
            start_time = times.min()
            end_time = times.max()
            duration = max(end_time - start_time, 0.1)  # 最小持续时间
            color = phase_colors_list[i % len(phase_colors_list)]
            fig.add_trace(
                go.Bar(
                    x=[duration],
                    y=[phase],
                    orientation='h',
                    base=start_time,
                    marker_color=color,
                    opacity=0.8,
                    name=phase,
                    showlegend=False,
                    hovertemplate=f"{phase}<br>开始: {start_time:.1f}分钟<br>结束: {end_time:.1f}分钟<br>持续: {duration:.1f}分钟<extra></extra>"
                ),
                row=2, col=1
            )
    
    # 4. 燃油消耗
    fuel_used = trajectory_df['fuel'].iloc[0] - trajectory_df['fuel']
    fig.add_trace(
        go.Scatter(
            x=time_min, y=fuel_used,
            fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.2)',
            line=dict(color='#9b59b6', width=2), 
            name='燃油',
            hovertemplate='时间: %{x:.1f}分钟<br>燃油消耗: %{y:.0f}kg<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 更新坐标轴
    fig.update_xaxes(title_text="时间 (分钟)", gridcolor='#eee')
    fig.update_yaxes(title_text="高度 (m)", row=1, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="速度 (节)", row=1, col=2, gridcolor='#eee')
    fig.update_yaxes(title_text="", row=2, col=1, gridcolor='#eee', categoryorder='array', categoryarray=ordered_phases)
    fig.update_yaxes(title_text="燃油消耗 (kg)", row=2, col=2, gridcolor='#eee')
    
    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=11),
        margin=dict(l=60, r=20, t=50, b=50)
    )
    
    return fig


def create_attitude_figure(trajectory_df):
    """创建姿态角图表 - 俯仰、滚转、航向"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('🎯 俯仰角 (Pitch)', '✈️ 滚转角 (Roll)', '🧭 航向 (Heading)'),
        shared_xaxes=True,
        vertical_spacing=0.12
    )
    
    time_min = trajectory_df['time'] / 60
    
    # 俯仰角
    fig.add_trace(
        go.Scatter(
            x=time_min, y=trajectory_df['pitch'],
            line=dict(color='#3498db', width=2),
            name='俯仰角',
            hovertemplate='时间: %{x:.1f}分钟<br>俯仰: %{y:.1f}°<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 滚转角
    fig.add_trace(
        go.Scatter(
            x=time_min, y=trajectory_df['roll'],
            line=dict(color='#e67e22', width=2),
            name='滚转角',
            hovertemplate='时间: %{x:.1f}分钟<br>滚转: %{y:.1f}°<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 航向
    fig.add_trace(
        go.Scatter(
            x=time_min, y=trajectory_df['heading'],
            line=dict(color='#27ae60', width=2),
            name='航向',
            hovertemplate='时间: %{x:.1f}分钟<br>航向: %{y:.0f}°<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="时间 (分钟)", row=3, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="俯仰 (°)", row=1, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="滚转 (°)", row=2, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="航向 (°)", row=3, col=1, gridcolor='#eee')
    
    fig.update_layout(
        height=450,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=11),
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_control_figure(trajectory_df):
    """创建控制输入图表 - 油门、目标俯仰、目标滚转"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('⛽ 油门 (Throttle)', '🎮 目标俯仰指令', '🎮 目标滚转指令'),
        shared_xaxes=True,
        vertical_spacing=0.12
    )
    
    time_min = trajectory_df['time'] / 60
    
    # 油门
    fig.add_trace(
        go.Scatter(
            x=time_min, y=trajectory_df['throttle'] * 100,
            fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='#e74c3c', width=2),
            name='油门',
            hovertemplate='时间: %{x:.1f}分钟<br>油门: %{y:.0f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 目标俯仰指令
    if 'target_pitch' in trajectory_df.columns:
        fig.add_trace(
            go.Scatter(
                x=time_min, y=trajectory_df['target_pitch'],
                line=dict(color='#9b59b6', width=2),
                name='目标俯仰',
                hovertemplate='时间: %{x:.1f}分钟<br>目标俯仰: %{y:.1f}°<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 目标滚转指令
    if 'target_roll' in trajectory_df.columns:
        fig.add_trace(
            go.Scatter(
                x=time_min, y=trajectory_df['target_roll'],
                line=dict(color='#1abc9c', width=2),
                name='目标滚转',
                hovertemplate='时间: %{x:.1f}分钟<br>目标滚转: %{y:.1f}°<extra></extra>'
            ),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="时间 (分钟)", row=3, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="油门 (%)", row=1, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="目标俯仰 (°)", row=2, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="目标滚转 (°)", row=3, col=1, gridcolor='#eee')
    
    fig.update_layout(
        height=450,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=11),
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_energy_figure(trajectory_df):
    """创建能量变化图表 - 势能、动能、总能量"""
    time_min = trajectory_df['time'] / 60
    
    # 计算能量（单位：MJ）
    g = 9.81  # 重力加速度
    mass = trajectory_df['mass']
    alt = trajectory_df['alt']
    tas = trajectory_df['tas']
    
    # 势能 Ep = mgh
    potential_energy = mass * g * alt / 1e6  # MJ
    
    # 动能 Ek = 0.5 * m * v^2
    kinetic_energy = 0.5 * mass * tas**2 / 1e6  # MJ
    
    # 总机械能
    total_energy = potential_energy + kinetic_energy
    
    fig = go.Figure()
    
    # 势能
    fig.add_trace(go.Scatter(
        x=time_min, y=potential_energy,
        fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db', width=2),
        name='势能 (Ep)',
        hovertemplate='时间: %{x:.1f}分钟<br>势能: %{y:.0f} MJ<extra></extra>'
    ))
    
    # 动能
    fig.add_trace(go.Scatter(
        x=time_min, y=kinetic_energy,
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.3)',
        line=dict(color='#e74c3c', width=2),
        name='动能 (Ek)',
        hovertemplate='时间: %{x:.1f}分钟<br>动能: %{y:.0f} MJ<extra></extra>'
    ))
    
    # 总能量
    fig.add_trace(go.Scatter(
        x=time_min, y=total_energy,
        line=dict(color='#2c3e50', width=3),
        name='总机械能',
        hovertemplate='时间: %{x:.1f}分钟<br>总能量: %{y:.0f} MJ<extra></extra>'
    ))
    
    fig.update_layout(
        title='⚡ 能量变化曲线',
        xaxis_title='时间 (分钟)',
        yaxis_title='能量 (MJ)',
        height=350,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333', size=11),
        xaxis=dict(gridcolor='#eee'),
        yaxis=dict(gridcolor='#eee'),
        margin=dict(l=60, r=20, t=80, b=40)
    )
    
    return fig


def get_route_info(route_name: str):
    """获取航线信息，返回推荐机型"""
    if not route_name:
        return "### 📍 航线信息\n*请选择航线*"
    
    route_idx = route_options.index(route_name)
    route = routes_df.iloc[route_idx]
    
    recommended = route.get('recommended_aircraft', '未知')
    distance = route['distance_km']
    
    return f"""### 📍 航线信息
- **距离**: {distance:.0f} km
- **推荐机型**: {recommended}
"""


def get_speed_range(aircraft_type: str):
    """获取机型的巡航速度范围，返回(min, max, optimal)"""
    if not aircraft_type:
        return 0.70, 0.90, 0.80
    
    ac = aircraft_df[aircraft_df['aircraft_type'] == aircraft_type].iloc[0]
    mach_min = ac.get('cruise_speed_mach_min', 0.74)
    mach_max = ac.get('cruise_speed_mach_max', 0.85)
    mach_opt = ac['cruise_speed_mach']
    
    return float(mach_min), float(mach_max), float(mach_opt)


def get_altitude_range(aircraft_type: str):
    """获取机型的巡航高度范围，返回(min, max, recommended)"""
    if not aircraft_type:
        return 10000, 45000, 35000
    
    ac = aircraft_df[aircraft_df['aircraft_type'] == aircraft_type].iloc[0]
    # 使用Service Ceiling作为最大高度
    alt_max = ac.get('service_ceiling_ft', 41000)
    # 推荐高度
    alt_rec = ac.get('cruise_alt_ft', 35000)
    # 最小高度定义为10000英尺或推荐高度的60%
    alt_min = 10000
    
    return float(alt_min), float(alt_max), float(alt_rec)


def update_speed_slider(aircraft_type: str):
    """更新巡航速度滑动条的范围"""
    mach_min, mach_max, mach_opt = get_speed_range(aircraft_type)
    
    # 获取机型信息
    ac = aircraft_df[aircraft_df['aircraft_type'] == aircraft_type].iloc[0]
    
    manuf = ac.get('manufacturer', 'N/A')
    pax = ac.get('typical_pax', 0)
    length = ac.get('length_m', 0)
    wingspan = ac.get('wingspan_m', 0)
    thrust = ac.get('max_thrust_n', 0)
    max_range = ac.get('max_range_km', 0)
    cruise_alt_ft = ac.get('cruise_alt_ft', 0)
    
    info_md = f"""
    ### ✈️ {aircraft_type} 详细参数
    *   **型号**: {aircraft_type}
    *   **制造商**: {manuf}
    *   **典型载客量**: {pax} 人
    *   **机身长度**: {length} m
    *   **翼展**: {wingspan} m
    *   **最大推力**: {thrust:,.0f} N
    *   **最大航程**: {max_range:,.0f} km
    *   **巡航高度**: {cruise_alt_ft:,.0f} ft
    """
    
    return gr.update(minimum=mach_min, maximum=mach_max, value=mach_opt, label=f"🚀 巡航马赫数 ({mach_min:.2f} - {mach_max:.2f})"), info_md


def update_altitude_slider(aircraft_type: str):
    """更新巡航高度滑动条的范围 (单位: 米)"""
    # 获取英尺数据
    alt_min_ft, alt_max_ft, alt_rec_ft = get_altitude_range(aircraft_type)
    
    # 转换为米
    alt_min_m = int(alt_min_ft * 0.3048)
    alt_max_m = int(alt_max_ft * 0.3048)
    alt_rec_m = int(alt_rec_ft * 0.3048)
    
    # 圆整到100米
    alt_min_m = (alt_min_m // 100) * 100
    alt_max_m = (alt_max_m // 100) * 100
    alt_rec_m = (alt_rec_m // 100) * 100
    
    return gr.update(minimum=alt_min_m, maximum=alt_max_m, value=alt_rec_m, step=100, label=f"🏔️ 巡航高度 ({alt_min_m} - {alt_max_m} m)")


def process_dataframe(df: pd.DataFrame, unit_system: str, time_format: str) -> pd.DataFrame:
    """处理数据帧：单位转换和时间格式化"""
    df_out = df.copy()
    
    # 1. 基础重命名（通用单位）
    rename_map = {
        'lat': 'latitude [deg]',
        'lon': 'longitude [deg]',
        'heading': 'heading [deg]',
        'pitch': 'pitch [deg]',
        'roll': 'roll [deg]',
        'target_pitch': 'target_pitch [deg]',
        'target_roll': 'target_roll [deg]',
        'throttle': 'throttle [0-1]',
        'flight_phase': 'flight_phase [-]'
    }
    
    # 2. 时间处理
    if time_format == "真实时间":
        now = datetime.datetime.now()
        # 将秒数转换为 timedelta 并加到当前时间
        df_out['time'] = df_out['time'].apply(lambda s: (now + datetime.timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S"))
        rename_map['time'] = 'time [YYYY-MM-DD HH:MM:SS]'
    else:
        # 默认模式：仿真时间（秒）
        rename_map['time'] = 'time [s]' 

    # 3. 单位转换
    if unit_system == "英制":
        # 公制 -> 英制
        # Alt: m -> ft
        df_out['alt'] = df_out['alt'] * 3.28084
        rename_map['alt'] = 'altitude [ft]'
        
        # Speed: m/s -> kts
        df_out['tas'] = df_out['tas'] * 1.94384
        rename_map['tas'] = 'true_airspeed [kts]'
        
        # Dist: m -> nm
        df_out['dist_to_dest'] = df_out['dist_to_dest'] / 1852.0
        rename_map['dist_to_dest'] = 'distance_to_dest [nm]'
        
        # Fuel/Mass: kg -> lbs
        df_out['fuel'] = df_out['fuel'] * 2.20462
        rename_map['fuel'] = 'fuel [lbs]'
        if 'mass' in df_out.columns:
            df_out['mass'] = df_out['mass'] * 2.20462
            rename_map['mass'] = 'mass [lbs]'
            
    else: # 公制 (默认)
        rename_map['alt'] = 'altitude [m]'
        rename_map['tas'] = 'true_airspeed [m/s]'
        rename_map['fuel'] = 'fuel [kg]'
        rename_map['mass'] = 'mass [kg]'
        # 距离转换为 km 更易读
        df_out['dist_to_dest'] = df_out['dist_to_dest'] / 1000.0
        rename_map['dist_to_dest'] = 'distance_to_dest [km]'
    
    # 应用重命名
    # 仅重命名存在的列
    final_rename = {k: v for k, v in rename_map.items() if k in df_out.columns}
    df_out = df_out.rename(columns=final_rename)
    
    return df_out


def run_simulation(route_name: str, aircraft_type: str, cruise_mach: float, cruise_alt_m: float, 
                   unit_system: str, time_format: str, progress=gr.Progress()):
    """运行模拟并返回可视化结果"""
    if not route_name or not aircraft_type:
        return None, None, None, None, None, "⚠️ 请选择航线和机型", None
    
    # 解析航线索引
    route_idx = route_options.index(route_name)
    route = routes_df.iloc[route_idx]
    
    # 将输入高度（米）转换为英尺供底层使用
    cruise_alt_ft = cruise_alt_m * 3.28084
    
    progress(0.1, desc="正在初始化模型...")
    
    # 生成轨迹（使用指定的巡航速度）
    progress(0.2, desc="正在生成飞行轨迹...")
    trajectory_df = generate_trajectory(aircraft_type, route, 
                                      cruise_speed_mach=cruise_mach,
                                      cruise_alt_ft=cruise_alt_ft)
    
    progress(0.6, desc="正在生成可视化...")
    
    # 创建可视化
    map_fig = create_map_figure(trajectory_df, route)
    analysis_fig = create_analysis_figure(trajectory_df)
    attitude_fig = create_attitude_figure(trajectory_df)
    control_fig = create_control_figure(trajectory_df)
    energy_fig = create_energy_figure(trajectory_df)
    
    progress(0.9, desc="正在计算统计数据...")
    
    # 生成统计信息
    total_time = trajectory_df['time'].max()
    fuel_used = trajectory_df['fuel'].iloc[0] - trajectory_df['fuel'].iloc[-1]
    max_alt = trajectory_df['alt'].max()
    max_speed = trajectory_df['tas'].max()
    
    # 获取飞机信息
    aircraft = aircraft_df[aircraft_df['aircraft_type'] == aircraft_type].iloc[0]
    mach_opt = aircraft['cruise_speed_mach']
    range_cat = aircraft.get('range_category', 'MEDIUM_HAUL')
    
    # 判断速度偏离最优值的程度
    speed_diff = abs(cruise_mach - mach_opt)
    if speed_diff < 0.01:
        speed_note = "✅ 最优巡航速度"
    elif cruise_mach < mach_opt:
        speed_note = "📉 经济巡航（较省油）"
    else:
        speed_note = "📈 高速巡航（较耗油）"

    # 判断高度偏离建议值的程度
    rec_alt_ft = aircraft['cruise_alt_ft']
    rec_alt_m = rec_alt_ft * 0.3048
    
    if abs(cruise_alt_m - rec_alt_m) < 150:
        alt_note = "✅ 建议高度"
    elif cruise_alt_m > rec_alt_m + 600:
        alt_note = "☁️ 较高高度"
    else:
        alt_note = "📉 较低高度"
    
    stats = f"""
## 📊 飞行统计

| 指标 | 数值 |
|:-----|-----:|
| **航线距离** | {route['distance_km']:.0f} km |
| **飞行时间** | {total_time/60:.1f} 分钟 |
| **最大高度** | {max_alt:.0f} m ({max_alt*3.28084:.0f} ft) |
| **设定巡航高度** | {cruise_alt_m:.0f} m ({cruise_alt_m*3.28084:.0f} ft) {alt_note} |
| **巡航马赫数** | {cruise_mach:.3f} ({speed_note}) |
| **最大速度** | {max_speed:.1f} m/s ({max_speed*1.944:.0f} 节) |
| **燃油消耗** | {fuel_used:.0f} kg |

### 💰 经济性分析
| 指标 | 数值 |
|:-----|-----:|
| **平均油耗** | {fuel_used / (total_time/60):.1f} kg/min |
| **每公里油耗** | {fuel_used / route['distance_km']:.2f} kg/km |
| **单座百公里油耗** | {(fuel_used / route['distance_km'] * 100) / aircraft['typical_pax']:.2f} kg/pax/100km |
"""
    
    progress(0.95, desc="正在保存文件...")
    # 保存 CSV 到临时文件
    csv_filename = f"flight_trajectory_{aircraft_type}_{route['origin_code']}-{route['dest_code']}.csv"
    
    # 根据用户选择处理 DataFrame (单位转换和时间格式)
    df_export = process_dataframe(trajectory_df, unit_system, time_format)
    
    # 创建临时文件，delete=False 确保 Gradio 可以读取（Gradio 会处理副本）
    # 使用 tempfile 生成一个临时路径，但保持我们想要的文件名后缀
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / csv_filename
    
    df_export.to_csv(temp_path, index=False)
    
    progress(1.0, desc="完成!")
    
    return map_fig, analysis_fig, attitude_fig, control_fig, energy_fig, stats, gr.File(value=str(temp_path), visible=True, label=f"📥 下载: {csv_filename}")


# 创建 Gradio 界面
demo = gr.Blocks(title="FlightSim")

with demo:
    gr.Markdown("""
    # ✈️FlightSim
    
    基于六自由度动力学模型的飞行轨迹生成与可视化工具。选择航线和机型，调整巡航速度，点击 **Run** 生成完整飞行轨迹。
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            route_dropdown = gr.Dropdown(
                choices=route_options,
                label="🛫 选择航线",
                value=route_options[0] if route_options else None,
                filterable=True
            )
            route_info = gr.Markdown("### 📍 航线信息\n*请选择航线*")
        with gr.Column(scale=2):
            aircraft_dropdown = gr.Dropdown(
                choices=aircraft_options,
                label="🛩️ 选择机型",
                value=aircraft_options[0] if aircraft_options else None
            )
            aircraft_info = gr.Markdown("*选择机型后显示参数范围*")
        with gr.Column(scale=2):
            # 初始化滑动条
            mach_min, mach_max, mach_opt = get_speed_range(aircraft_options[0] if aircraft_options else None)
            
            # 高度范围（米）
            alt_min_ft, alt_max_ft, alt_rec_ft = get_altitude_range(aircraft_options[0] if aircraft_options else None)
            alt_min_m = int(alt_min_ft * 0.3048 / 100) * 100
            alt_max_m = int(alt_max_ft * 0.3048 / 100) * 100
            alt_rec_m = int(alt_rec_ft * 0.3048 / 100) * 100
            
            cruise_slider = gr.Slider(
                minimum=mach_min,
                maximum=mach_max,
                value=mach_opt,
                step=0.005,
                label=f"🚀 巡航马赫数 ({mach_min:.2f} - {mach_max:.2f})"
            )
            
            alt_slider = gr.Slider(
                minimum=alt_min_m,
                maximum=alt_max_m,
                value=alt_rec_m,
                step=100,
                label=f"🏔️ 巡航高度 ({alt_min_m} - {alt_max_m} m)"
            )
            
            with gr.Row():
                unit_radio = gr.Radio(
                    choices=["公制", "英制"],
                    value="公制",
                    label="单位",
                    scale=1,
                    container=False
                )
                time_radio = gr.Radio(
                    choices=["仿真时间", "真实时间"],
                    value="仿真时间",
                    label="时间",
                    scale=1,
                    container=False
                )
                
        with gr.Column(scale=1):
            run_btn = gr.Button("🚀 Run", variant="primary", size="lg")
            download_file = gr.File(label="📥 下载 CSV", visible=False)
    
    with gr.Tabs():
        with gr.TabItem("🌍 轨迹概览"):
            with gr.Row():
                with gr.Column(scale=3):
                    map_plot = gr.Plot(label="飞行轨迹地图")
                with gr.Column(scale=2):
                    stats_md = gr.Markdown("*选择航线和机型后点击 Run 开始模拟*")
        
        with gr.TabItem("📈 详细数据"):
            analysis_plot = gr.Plot(label="航迹详细分析（高度/速度/阶段/燃油）")
            
        with gr.TabItem("✈️ 动力学分析"):
            with gr.Row():
                attitude_plot = gr.Plot(label="姿态角变化")
                control_plot = gr.Plot(label="控制输入")
            energy_plot = gr.Plot(label="能量变化")
    
    # 事件绑定
    route_dropdown.change(
        fn=get_route_info,
        inputs=[route_dropdown],
        outputs=[route_info]
    )
    
    aircraft_dropdown.change(
        fn=lambda ac: (update_speed_slider(ac)[0], update_speed_slider(ac)[1], update_altitude_slider(ac)),
        inputs=[aircraft_dropdown],
        outputs=[cruise_slider, aircraft_info, alt_slider]
    )
    
    run_btn.click(
        fn=run_simulation,
        inputs=[route_dropdown, aircraft_dropdown, cruise_slider, alt_slider, unit_radio, time_radio],
        outputs=[map_plot, analysis_plot, attitude_plot, control_plot, energy_plot, stats_md, download_file]
    )



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


