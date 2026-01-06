
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
import random
from tqdm import tqdm
import json
from datetime import datetime
import multiprocessing
from functools import partial

# ============================================================================
# 用户配置区域 (User Configuration)
# ============================================================================

# 1. 路由设置
# 需要处理的航线数量上限 (遍历前N条航线)
NUM_ROUTES_TO_PROCESS = 300

# 2. 仿真设置
# 仿真时间步长 (秒)
DT = 0.1 
# 最大仿真时长倍数 (基于预计飞行时间的倍数，防止无限循环)
MAX_DURATION_MULTIPLIER = 1.5

# 3. 扰动/过程噪声设置 (随机数范围 - 参考 Gradio Demo)
# -----------------------------------------------------------
# 风场湍流强度范围 (m/s)
# 对应 Dryden 模型风速标准差: 
# 轻度 ~1 m/s, 中度 ~3 m/s, 重度 ~6 m/s, 极端 ~20 m/s
# 脚本将在此范围内随机取值，并在后续除以 20.0 进行归一化以匹配 NoiseConfig
WIND_SPEED_RANGE_MS = [1.0, 15.0]

# 气动摄动强度范围 (无量纲)
# 湍流引起的气动力扰动，Gradio建议 <= 0.3
# 建议范围: 0.0 - 0.2
AERO_PERTURBATION_RANGE = [0.0, 0.2]

# 4. 量测噪声设置
# -----------------------------------------------------------
# 定义要生成的噪声类型，每一类将单独生成一个文件夹
# 可选类型: ["white", "flicker", "drift", "colored", "timevar"]
TARGET_NOISE_TYPES = ["white", "flicker", "drift", "colored", "timevar"]

# 基础噪声强度范围 (Sigma归一化系数 0-1)
# 0.1 表示较小噪声, 1.0 表示最大定义噪声
NOISE_INTENSITY_RANGE = [0.1, 0.8]

# 各类噪声的特定参数随机范围 (参考 Gradio Demo)
NOISE_PARAM_RANGES = {
    # 闪烁噪声: 偶尔出现大幅度跳变
    "flicker": {
        "prob_range": [0.01, 0.2],   # 闪烁概率 (0-1)
        "scale_range": [2.0, 10.0]   # 闪烁幅度倍数 (0-50)
    },
    # 漂移噪声: 随时间累积误差
    "drift": {
        "rate_range": [0.001, 0.01]  # 漂移率 (0-0.02)
    },
    # 有色噪声: 时间相关噪声
    "colored": {
        "alpha_range": [0.6, 0.95]   # 相关系数 (0.5-0.99)
    },
    # 时变噪声: 噪声强度随时间周期变化
    "timevar": {
        "period_range": [50.0, 500.0], # 变化周期 (10-1000)
        "amp_range": [0.5, 3.0]        # 变化幅度 (0-5)
    }
}

# 5. 输出设置
# 数据集保存的根目录
OUTPUT_DIR_ROOT = "/home/b220/share2/dataset/flightsim/v1"

# 6. 并行设置
# 使用的CPU核心数 (None表示使用所有可用核心)
NUM_CORES = 20 

# ============================================================================
# 脚本逻辑 (Script Logic)
# ============================================================================

# 添加 src 到 python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from flightsim.sixdof import SixDOFModel
    from flightsim.autopilot import create_autopilot, FlightPhase
    from flightsim.noise import NoiseConfig, NoiseManager
except ImportError as e:
    print(f"Error importing flightsim modules: {e}")
    print("Please ensure you are running this script from the project root or tools directory.")
    sys.exit(1)

# 配置日志 (Main process only)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_random_noise_config(noise_type):
    """
    根据指定的噪声类型生成随机配置参数
    """
    # 1. 基础过程噪声 (每条航迹随机)
    # 风速 (m/s) -> 归一化 (0-1)
    wind_speed = random.uniform(WIND_SPEED_RANGE_MS[0], WIND_SPEED_RANGE_MS[1])
    wind_intensity_norm = wind_speed / 20.0
    
    aero_pert = random.uniform(AERO_PERTURBATION_RANGE[0], AERO_PERTURBATION_RANGE[1])
    
    # 2. 量测噪声强度
    # 假设IMU和GPS噪声强度具有一定的相关性，设定为相近的随机值
    base_intensity = random.uniform(NOISE_INTENSITY_RANGE[0], NOISE_INTENSITY_RANGE[1])
    # 增加一点随机扰动
    imu_intensity = np.clip(base_intensity + random.uniform(-0.1, 0.1), 0.0, 1.0)
    gps_intensity = np.clip(base_intensity + random.uniform(-0.1, 0.1), 0.0, 1.0)
    
    # 3. 特定类型参数
    params = {}
    
    # 获取特定类型的随机参数
    if noise_type == "flicker":
        params["flicker_prob"] = random.uniform(*NOISE_PARAM_RANGES["flicker"]["prob_range"])
        params["flicker_scale"] = random.uniform(*NOISE_PARAM_RANGES["flicker"]["scale_range"])
    elif noise_type == "drift":
        params["drift_rate"] = random.uniform(*NOISE_PARAM_RANGES["drift"]["rate_range"])
    elif noise_type == "colored":
        params["colored_alpha"] = random.uniform(*NOISE_PARAM_RANGES["colored"]["alpha_range"])
    elif noise_type == "timevar":
        params["timevar_period"] = random.uniform(*NOISE_PARAM_RANGES["timevar"]["period_range"])
        params["timevar_amp"] = random.uniform(*NOISE_PARAM_RANGES["timevar"]["amp_range"])
        
    # 构建 NoiseConfig
    config = NoiseConfig(
        # 过程噪声
        wind_intensity=wind_intensity_norm,
        aero_perturbation=aero_pert,
        
        # IMU噪声
        imu_noise=imu_intensity,
        imu_noise_type=noise_type,
        imu_flicker_prob=params.get("flicker_prob", 0.1),
        imu_flicker_scale=params.get("flicker_scale", 5.0),
        imu_drift_rate=params.get("drift_rate", 0.002),
        imu_colored_alpha=params.get("colored_alpha", 0.9),
        imu_timevar_period=params.get("timevar_period", 100.0),
        imu_timevar_amp=params.get("timevar_amp", 1.0),
        
        # GPS噪声
        gps_noise=gps_intensity,
        gps_noise_type=noise_type,
        gps_flicker_prob=params.get("flicker_prob", 0.1),
        gps_flicker_scale=params.get("flicker_scale", 5.0),
        gps_drift_rate=params.get("drift_rate", 0.002),
        gps_colored_alpha=params.get("colored_alpha", 0.9),
        gps_timevar_period=params.get("timevar_period", 100.0),
        gps_timevar_amp=params.get("timevar_amp", 1.0)
    )
    
    # 记录用于 Metadata 的配置字典
    meta_config = {
        "process_noise": {
            "wind_speed_ms": wind_speed,
            "wind_intensity_norm": wind_intensity_norm,
            "aero_perturbation": aero_pert
        },
        "measurement_noise": {
            "type": noise_type,
            "imu_intensity": imu_intensity,
            "gps_intensity": gps_intensity,
            "specific_params": params
        }
    }
    
    return config, meta_config

def check_trajectory_quality(df, route_code):
    """
    检查航迹质量: 是否闭合, 下降时间是否过长
    返回错误列表，为空表示通过
    """
    issues = []
    
    if df.empty:
        return ["Empty trajectory"]

    # 1. 检查闭合性 (Closure)
    # 判据: 最终高度 < 50米 且 最终阶段为 ROLLOUT 或 TOUCHDOWN
    last_row = df.iloc[-1]
    last_alt = last_row['alt']
    last_phase = last_row['phase']
    
    if last_alt > 50.0:
        issues.append(f"Not Closed (Final Alt: {last_alt:.1f}m, Phase: {last_phase})")
    elif last_phase not in ['ROLLOUT', 'TOUCHDOWN', 'TAXI']:
        issues.append(f"Abnormal End Phase (Phase: {last_phase})")

    # 2. 检查下降时间 (Descent Duration)
    # 判据: 所有包含 'DESCENT' 或 'APPROACH' 的阶段总时长
    # 一般下降耗时 20-30分钟，如果超过 50分钟 (3000秒) 视为过长
    descent_mask = df['phase'].str.contains('DESCENT|APPROACH')
    if descent_mask.any():
        descent_times = df.loc[descent_mask, 'time']
        descent_start = descent_times.iloc[0]
        descent_end = descent_times.iloc[-1]
        descent_duration = descent_end - descent_start
        
        if descent_duration > 3000:
            issues.append(f"Excessive Descent Time ({descent_duration:.1f}s = {descent_duration/60:.1f}min)")
    
    return issues

def run_simulation_task(task_args):
    """
    运行单次仿真并保存数据 (Wrapper for Parallel Execution)
    Returns: (Success, Filename, ValidationIssues)
    """
    route, aircraft_type, noise_type, output_dir = task_args
    
    try:
        # 生成随机噪声配置
        noise_config, meta_config_dict = generate_random_noise_config(noise_type)
        
        # 1. 准备路径参数
        origin = (route['origin_lat'], route['origin_lon'])
        dest = (route['dest_lat'], route['dest_lon'])
        
        # 解析航路点
        waypoints = [origin]
        for i in range(1, 11):
            lat_key = f'waypoint{i}_lat'
            lon_key = f'waypoint{i}_lon'
            if pd.notna(route.get(lat_key)) and pd.notna(route.get(lon_key)):
                waypoints.append((route[lat_key], route[lon_key]))
        waypoints.append(dest)
        
        # 2. 初始化模型
        model = SixDOFModel(aircraft_type, 
                           start_lat=origin[0], 
                           start_lon=origin[1], 
                           start_alt=10.0, 
                           start_heading=0.0, 
                           dt=DT)
        
        noise_manager = NoiseManager(noise_config, model.dt)
        model.noise_obj = noise_config
        model.noise_manager = noise_manager
        
        # 3. 初始化自动驾驶仪
        autopilot = create_autopilot(model)
        autopilot.load_route(waypoints, departure_alt=10.0)
        
        # 4. 运行仿真
        dist_km = route['distance_km']
        max_duration = (dist_km / 600.0) * 3600 * MAX_DURATION_MULTIPLIER
        t = 0.0
        
        # 数据记录
        data_records = []
        
        while t < max_duration:
            # 计算速度分量
            v_horiz = model.tas * np.cos(model.gamma)
            vn = v_horiz * np.cos(np.radians(model.heading))
            ve = v_horiz * np.sin(np.radians(model.heading))
            vd = -model.v_vertical 
            
            # 获取真实状态
            true_state = {
                "time": t,
                "lat": model.lat,
                "lon": model.lon,
                "alt": model.alt,
                "vn": vn,
                "ve": ve,
                "vd": vd,
                "roll": model.roll,
                "pitch": model.pitch,
                "heading": model.heading,
                "phase": autopilot.phase.name
            }
            
            # 生成观测值
            meas_lat, meas_lon, meas_alt, meas_vel = noise_manager.apply_gps_noise(
                model.lat, model.lon, model.alt, np.array([vn, ve, vd])
            )
            meas_pitch, meas_roll, meas_heading = noise_manager.apply_attitude_noise(
                model.pitch, model.roll, model.heading
            )
            
            meas_state = {
                "meas_lat": meas_lat,
                "meas_lon": meas_lon,
                "meas_alt": meas_alt,
                "meas_vn": meas_vel[0],
                "meas_ve": meas_vel[1],
                "meas_vd": meas_vel[2],
                "meas_roll": meas_roll,
                "meas_pitch": meas_pitch,
                "meas_heading": meas_heading
            }
            
            # 合并记录
            record = {**true_state, **meas_state}
            data_records.append(record)
            
            # 更新步骤
            throttle, pitch_cmd, roll_cmd = autopilot.update()
            model.update(throttle, pitch_cmd, roll_cmd)
            
            # 终止条件
            if autopilot.phase in [FlightPhase.TOUCHDOWN, FlightPhase.ROLLOUT] and model.gs < 40.0:
                break
            if model.alt < 0:
                break
            
            t += model.dt

        # 5. 保存文件
        origin_code = route.get('origin_code', 'UNKNOWN')
        dest_code = route.get('dest_code', 'UNKNOWN')
        route_code = f"{origin_code}-{dest_code}"
        
        if origin_code == 'UNKNOWN' or dest_code == 'UNKNOWN':
             route_code = route['route_name'].replace(" ", "_").replace("/", "-")
        
        filename = f"{route_code}_{aircraft_type}_{noise_type}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(data_records)
        
        # 6. 质量检查
        quality_issues = check_trajectory_quality(df, route_code)
        # Note: Writing to log file is moved to main process to avoid race conditions
        
        df.to_csv(filepath, index=False)
        
        # 同时保存元数据
        meta_filepath = filepath.replace(".csv", ".json")
        metadata = {
            "route_code": route_code,
            "aircraft": aircraft_type,
            "duration": t,
            "noise_category": noise_type,
            "config": meta_config_dict,
            "validation_passed": len(quality_issues) == 0,
            "validation_issues": quality_issues
        }
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        summary_info = {
            "filename": filename,
            "origin_code": origin_code,
            "dest_code": dest_code,
            "aircraft": aircraft_type,
            "wind_speed_ms": meta_config_dict['process_noise']['wind_speed_ms'],
            "aero_perturbation": meta_config_dict['process_noise']['aero_perturbation'],
            "noise_type": noise_type,
            "imu_noise_intensity": meta_config_dict['measurement_noise']['imu_intensity'],
            "gps_noise_intensity": meta_config_dict['measurement_noise']['gps_intensity']
        }
            
        return True, filename, quality_issues, summary_info

    except Exception as e:
        # 打印错误但不中断整个过程
        return False, f"{route.get('route_name', 'Unknown')}_{aircraft_type}", [str(e)], None

def main():
    print(f"=== 开始生成航迹数据集 (Parallel) ===")
    print(f"计划处理航线数: {NUM_ROUTES_TO_PROCESS}")
    print(f"目标噪声类型: {TARGET_NOISE_TYPES}")
    print(f"过程噪声(风)范围: {WIND_SPEED_RANGE_MS} m/s")
    print(f"过程噪声气动摄动范围: {AERO_PERTURBATION_RANGE}")
    
    # 1. 加载航线数据
    waypoints_file = PROJECT_ROOT / "src/flightsim/data/waypoints.csv"
    if not waypoints_file.exists():
        print(f"错误: 找不到文件 {waypoints_file}")
        return
        
    routes_df = pd.read_csv(waypoints_file)
    total_routes = len(routes_df)
    routes_to_process = routes_df.head(min(total_routes, NUM_ROUTES_TO_PROCESS))
    
    # 2. 准备并行任务列表
    tasks = []
    
    # 创建所有需要的目录
    for noise_type in TARGET_NOISE_TYPES:
        current_output_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR_ROOT, noise_type)
        os.makedirs(current_output_dir, exist_ok=True)
        
        # 为每条航线和每个机型创建任务
        for idx, route in routes_to_process.iterrows():
            rec_aircraft_str = str(route['recommended_aircraft'])
            if pd.isna(rec_aircraft_str) or not rec_aircraft_str.strip():
                continue
                
            rec_aircraft_list = [a.strip() for a in rec_aircraft_str.split(',') if a.strip()]
            
            for aircraft_type in rec_aircraft_list:
                tasks.append((route, aircraft_type, noise_type, current_output_dir))
    
    print(f"总任务数: {len(tasks)}")
    
    # 3. 并行执行任务
    # 使用 Python 的 multiprocessing.Pool
    # 如果 NUM_CORES 为 None，默认使用 os.cpu_count()
    cpu_count = multiprocessing.cpu_count() if NUM_CORES is None else NUM_CORES
    print(f"启动并行池 (Cores: {cpu_count})...")
    
    success_count = 0
    fail_count = 0
    
    validation_log_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR_ROOT, "validation_log.txt")
    summary_list = []
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # 使用 tqdm 显示总体进度
        with tqdm(total=len(tasks), desc="Processing Tasks") as pbar:
            for success, filename, issues, summary_info in pool.imap_unordered(run_simulation_task, tasks):
                
                if success:
                    success_count += 1
                    # 检查是否有质量问题需要记录
                    if issues:
                        with open(validation_log_path, "a", encoding='utf-8') as f:
                            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for issue in issues:
                                f.write(f"[{timestamp_str}] [FAIL] {filename}: {issue}\n")
                    else:
                        # 只有通过验证的才加入统计列表
                        if summary_info:
                            summary_list.append(summary_info)
                else:
                    fail_count += 1
                    logger.error(f"Task failed for {filename}: {issues}")
                
                pbar.update(1)
    
    # 保存汇总统计 CSV
    if summary_list:
        summary_csv_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR_ROOT, "total_list.csv")
        summary_df = pd.DataFrame(summary_list)
        # 调整列顺序
        columns_order = ["filename", "origin_code", "dest_code", "aircraft", 
                         "wind_speed_ms", "aero_perturbation", 
                         "noise_type", "imu_noise_intensity", "gps_noise_intensity"]
        # 确保所有列存在
        existing_cols = [c for c in columns_order if c in summary_df.columns]
        summary_df = summary_df[existing_cols]
        
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"汇总统计已保存至: {summary_csv_path}")
        
    print(f"\n=== 所有任务完成 ===")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"结果已保存至: {os.path.join(PROJECT_ROOT, OUTPUT_DIR_ROOT)}")

if __name__ == "__main__":
    main()
