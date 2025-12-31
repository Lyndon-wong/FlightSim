"""
航线和轨迹可视化工具
整合了航线可视化、轨迹分析等功能
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_routes(csv_path='waypoints.csv', output_html='routes_map.html', max_routes=100):
    """
    可视化航线网络
    
    Args:
        csv_path: 航线CSV文件路径
        output_html: 输出HTML文件路径
        max_routes: 最多显示的航线数量
    """
    df = pd.read_csv(csv_path)
    
    # 采样航线
    if len(df) > max_routes:
        df_sample = df.sample(n=max_routes, random_state=42)
    else:
        df_sample = df
    
    fig = go.Figure()
    
    # 颜色映射
    category_colors = {
        '中国': '#FF6B6B', '亚洲': '#4ECDC4', '跨太平洋': '#45B7D1',
        '跨大西洋': '#FFA07A', '亚欧': '#98D8C8', '美国': '#F7DC6F',
        '欧洲': '#BB8FCE', '其他': '#85C1E2'
    }
    
    # 绘制每条航线
    for _, route in df_sample.iterrows():
        lats = [route['origin_lat']]
        lons = [route['origin_lon']]
        
        for i in range(1, 11):
            wp_lat = route.get(f'waypoint{i}_lat')
            wp_lon = route.get(f'waypoint{i}_lon')
            if pd.notna(wp_lat) and pd.notna(wp_lon):
                lats.append(wp_lat)
                lons.append(wp_lon)
        
        lats.append(route['dest_lat'])
        lons.append(route['dest_lon'])
        
        color = category_colors.get(route['category'], '#999999')
        
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats, mode='lines',
            line=dict(width=1.5, color=color),
            name=route['route_name'],
            hovertext=f"{route['route_name']}<br>{route['distance_km']:.0f}km"
        ))
    
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
        showocean=True, oceancolor='rgb(230, 245, 255)',
    )
    
    fig.update_layout(
        title=f"全球航线网络 ({len(df_sample)}条)",
        showlegend=False, height=600
    )
    
    fig.write_html(output_html)
    print(f"✓ 航线地图已保存到: {output_html}")
    
    return fig


def analyze_routes(csv_path='waypoints.csv'):
    """分析航线数据统计"""
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("航线数据统计")
    print("=" * 60)
    print(f"总航线数: {len(df)}")
    print(f"\n按类别统计:")
    print(df['category'].value_counts())
    
    print(f"\n距离统计:")
    print(f"  最短: {df['distance_km'].min():.0f} km")
    print(f"  最长: {df['distance_km'].max():.0f} km")
    print(f"  平均: {df['distance_km'].mean():.0f} km")
    print(f"  中位数: {df['distance_km'].median():.0f} km")
    
    # 航路点数量统计
    waypoint_counts = []
    for _, row in df.iterrows():
        count = sum(1 for i in range(1, 11) if pd.notna(row.get(f'waypoint{i}_lat')))
        waypoint_counts.append(count)
    
    print(f"\n航路点统计:")
    print(f"  最少: {min(waypoint_counts)}")
    print(f"  最多: {max(waypoint_counts)}")
    print(f"  平均: {np.mean(waypoint_counts):.1f}")


def visualize_trajectory(trajectory_csv, output_file='trajectory_plot.png'):
    """可视化单条轨迹"""
    df = pd.read_csv(trajectory_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 高度剖面
    axes[0, 0].plot(df['time']/60, df['alt'], 'b-', linewidth=1.5)
    axes[0, 0].fill_between(df['time']/60, 0, df['alt'], alpha=0.3)
    axes[0, 0].set_xlabel('时间 (分钟)')
    axes[0, 0].set_ylabel('高度 (m)')
    axes[0, 0].set_title('高度剖面')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 速度曲线
    axes[0, 1].plot(df['time']/60, df['tas']*1.944, 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('时间 (分钟)')
    axes[0, 1].set_ylabel('速度 (节)')
    axes[0, 1].set_title('速度曲线')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 飞行阶段
    if 'flight_phase' in df.columns:
        phases = df['flight_phase'].unique()
        phase_colors = plt.cm.tab20(np.linspace(0, 1, len(phases)))
        for i, phase in enumerate(phases):
            mask = df['flight_phase'] == phase
            times = df[mask]['time'] / 60
            if len(times) > 0:
                axes[1, 0].barh(phase, times.max() - times.min(), 
                              left=times.min(), color=phase_colors[i], alpha=0.7)
        axes[1, 0].set_xlabel('时间 (分钟)')
        axes[1, 0].set_title('飞行阶段')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 燃油消耗
    if 'fuel' in df.columns:
        fuel_used = df['fuel'].iloc[0] - df['fuel']
        axes[1, 1].plot(df['time']/60, fuel_used, 'm-', linewidth=1.5)
        axes[1, 1].fill_between(df['time']/60, 0, fuel_used, alpha=0.3, color='m')
        axes[1, 1].set_xlabel('时间 (分钟)')
        axes[1, 1].set_ylabel('燃油消耗 (kg)')
        axes[1, 1].set_title('燃油消耗')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 轨迹图已保存到: {output_file}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='航线和轨迹可视化工具')
    parser.add_argument('--routes', action='store_true', help='可视化航线网络')
    parser.add_argument('--analyze', action='store_true', help='分析航线统计')
    parser.add_argument('--trajectory', type=str, help='可视化轨迹文件')
    parser.add_argument('--input', default='waypoints.csv', help='输入航线文件')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    if args.routes:
        output = args.output or 'routes_map.html'
        visualize_routes(args.input, output)
    elif args.analyze:
        analyze_routes(args.input)
    elif args.trajectory:
        output = args.output or 'trajectory_plot.png'
        visualize_trajectory(args.trajectory, output)
    else:
        print("使用示例:")
        print("  python visualize.py --routes          # 可视化航线网络")
        print("  python visualize.py --analyze         # 分析航线统计")
        print("  python visualize.py --trajectory trajectory.csv  # 可视化轨迹")

