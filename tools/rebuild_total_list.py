#!/usr/bin/env python3
"""
重新生成数据集汇总文件 total_list.csv

该脚本遍历 ~/share2/dataset/flightsim/v1/ 下的所有噪声类型目录，
从JSON元数据文件中提取信息，生成汇总的total_list.csv文件。

只包含通过验证的数据（validation_passed == true）。
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 配置
DATASET_ROOT = Path.home() / "share2/dataset/flightsim/v1"
OUTPUT_CSV = DATASET_ROOT / "total_list.csv"
NOISE_TYPES = ["white", "flicker", "drift", "colored", "timevar"]

def extract_info_from_json(json_path):
    """
    从JSON元数据文件中提取需要的信息
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        dict: 包含提取信息的字典，如果验证未通过则返回None
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 只处理通过验证的数据
        if not data.get("validation_passed", False):
            return None
        
        # 从route_code分离起点和终点
        route_code = data.get("route_code", "UNKNOWN-UNKNOWN")
        parts = route_code.split("-")
        origin_code = parts[0] if len(parts) >= 1 else "UNKNOWN"
        dest_code = parts[1] if len(parts) >= 2 else "UNKNOWN"
        
        # 构建CSV文件名（从JSON文件名推导）
        json_filename = os.path.basename(json_path)
        csv_filename = json_filename.replace(".json", ".csv")
        
        # 提取配置信息
        config = data.get("config", {})
        process_noise = config.get("process_noise", {})
        measurement_noise = config.get("measurement_noise", {})
        
        return {
            "filename": csv_filename,
            "origin_code": origin_code,
            "dest_code": dest_code,
            "aircraft": data.get("aircraft", "UNKNOWN"),
            "wind_speed_ms": process_noise.get("wind_speed_ms", 0.0),
            "aero_perturbation": process_noise.get("aero_perturbation", 0.0),
            "noise_type": data.get("noise_category", measurement_noise.get("type", "unknown")),
            "imu_noise_intensity": measurement_noise.get("imu_intensity", 0.0),
            "gps_noise_intensity": measurement_noise.get("gps_intensity", 0.0)
        }
    
    except Exception as e:
        print(f"警告: 无法处理文件 {json_path}: {e}")
        return None

def main():
    print("=== 开始重新生成数据集汇总文件 ===")
    print(f"数据集根目录: {DATASET_ROOT}")
    print(f"输出文件: {OUTPUT_CSV}")
    print()
    
    # 收集所有JSON文件
    all_json_files = []
    for noise_type in NOISE_TYPES:
        noise_dir = DATASET_ROOT / noise_type
        if not noise_dir.exists():
            print(f"警告: 目录不存在 {noise_dir}")
            continue
        
        # 查找所有JSON文件
        json_files = list(noise_dir.glob("*.json"))
        all_json_files.extend(json_files)
        print(f"发现 {len(json_files)} 个JSON文件在 {noise_type}/ 目录")
    
    print(f"\n总共发现 {len(all_json_files)} 个JSON文件")
    print("开始提取信息...\n")
    
    # 提取信息
    records = []
    skipped_count = 0
    
    for json_path in tqdm(all_json_files, desc="处理JSON文件"):
        info = extract_info_from_json(json_path)
        if info is not None:
            records.append(info)
        else:
            skipped_count += 1
    
    # 创建DataFrame
    if not records:
        print("\n错误: 没有找到有效的数据记录！")
        return
    
    df = pd.DataFrame(records)
    
    # 调整列顺序（与原始脚本一致）
    columns_order = [
        "filename", "origin_code", "dest_code", "aircraft",
        "wind_speed_ms", "aero_perturbation",
        "noise_type", "imu_noise_intensity", "gps_noise_intensity"
    ]
    
    # 确保所有列存在
    existing_cols = [c for c in columns_order if c in df.columns]
    df = df[existing_cols]
    
    # 保存到CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n=== 汇总完成 ===")
    print(f"总处理文件数: {len(all_json_files)}")
    print(f"有效记录数: {len(records)}")
    print(f"跳过记录数: {skipped_count} (未通过验证)")
    print(f"\n汇总文件已保存至: {OUTPUT_CSV}")
    print(f"文件大小: {OUTPUT_CSV.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    main()
