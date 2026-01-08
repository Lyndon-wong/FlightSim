# FlightSim 航迹数据集文档

## 概述

FlightSim 航迹数据集是一个大规模的飞行仿真数据集，包含多种噪声条件下的航班3D轨迹数据。数据集主要用于训练和评估容噪的航迹预测模型。（目前航线还是最优大圆航线，所以仅供生成训练数据用。）

**数据集版本**: v1  
**生成日期**: 2026-01-07 ~ 2026-01-08  
---

## 数据集统计

### 整体规模

| 指标 | 数值 |
|------|------|
| **总记录数** | 5,031 条 |
| **唯一航线数** | 279 条 |
| **飞机型号数** | 14 种 |
| **噪声类型数** | 5 种 |
| **数据文件总数** | 10,062 个（5,031 CSV + 5,031 JSON） |
| **数据集大小** | ~13 GB |
| **数据质量通过率** | 92.4% (5,031/5,444) |

### 噪声类型分布

| 噪声类型 | 记录数 | 占比 | 说明 |
|----------|--------|------|------|
| **white** | 1,104 | 21.9% | 白噪声（基础高斯噪声） |
| **flicker** | 1,104 | 21.9% | 闪烁噪声（偶发大幅度跳变） |
| **drift** | 1,104 | 21.9% | 漂移噪声（累积误差） |
| **colored** | 1,083 | 21.5% | 有色噪声（时间相关） |
| **timevar** | 636 | 12.6% | 时变噪声（周期性强度变化） |

### 飞机型号分布（Top 10）

| 飞机型号 | 记录数 | 占比 | 类别 |
|----------|--------|------|------|
| B737-800 | 645 | 12.8% | 窄体中短程 |
| A320-200 | 645 | 12.8% | 窄体中短程 |
| A321-200 | 482 | 9.6% | 窄体中短程 |
| B757-200 | 482 | 9.6% | 窄体中长程 |
| A350-900 | 357 | 7.1% | 宽体长程 |
| B777-300ER | 356 | 7.1% | 宽体长程 |
| B787-9 | 354 | 7.0% | 宽体长程 |
| A380-800 | 334 | 6.6% | 超大型宽体 |
| CRJ-900 | 265 | 5.3% | 支线客机 |
| ERJ-145 | 265 | 5.3% | 支线客机 |

---

## 目录结构

```
~/share2/dataset/flightsim/v1/
├── white/                      # 白噪声数据
│   ├── *.csv                   # 航迹数据文件
│   └── *.json                  # 元数据文件
├── flicker/                    # 闪烁噪声数据
│   ├── *.csv
│   └── *.json
├── drift/                      # 漂移噪声数据
│   ├── *.csv
│   └── *.json
├── colored/                    # 有色噪声数据
│   ├── *.csv
│   └── *.json
├── timevar/                    # 时变噪声数据
│   ├── *.csv
│   └── *.json
├── total_list.csv              # 数据集汇总索引
└── validation_log.txt          # 质量验证日志
```

---

## 文件格式说明

### 1. 航迹数据文件 (*.csv)

每个CSV文件对应一条完整的航迹，包含以下字段：

#### 时间与飞行阶段
- `time` - 仿真时间（秒）
- `phase` - 飞行阶段（TAKEOFF/CLIMB/CRUISE/DESCENT/APPROACH/FINAL/FLARE/TOUCHDOWN/ROLLOUT）

#### 真实状态（Ground Truth）
- `lat` - 纬度（度）
- `lon` - 经度（度）
- `alt` - 高度（米）
- `vn` - 北向速度（m/s）
- `ve` - 东向速度（m/s）
- `vd` - 垂直速度（m/s，向下为正）
- `roll` - 滚转角（度）
- `pitch` - 俯仰角（度）
- `heading` - 航向角（度）

#### 观测值（含噪声）
- `meas_lat` - 观测纬度
- `meas_lon` - 观测经度
- `meas_alt` - 观测高度
- `meas_vn` - 观测北向速度
- `meas_ve` - 观测东向速度
- `meas_vd` - 观测垂直速度
- `meas_roll` - 观测滚转角
- `meas_pitch` - 观测俯仰角
- `meas_heading` - 观测航向角

**数据维度**: 每条航迹包含数千到数万个时间步（取决于航程长度）  
**采样频率**: 10 Hz (dt = 0.1s)

### 2. 元数据文件 (*.json)

每个JSON文件包含对应航迹的配置和验证信息：

```json
{
    "route_code": "PEK-PVG",
    "aircraft": "A320-200",
    "duration": 5432.1,
    "noise_category": "white",
    "config": {
        "process_noise": {
            "wind_speed_ms": 5.01,
            "wind_intensity_norm": 0.25,
            "aero_perturbation": 0.05
        },
        "measurement_noise": {
            "type": "white",
            "imu_intensity": 0.64,
            "gps_intensity": 0.68,
            "specific_params": {}
        }
    },
    "validation_passed": true,
    "validation_issues": []
}
```

### 3. 数据集汇总索引 (total_list.csv)

整个数据集的索引文件，包含以下列：

| 字段 | 说明 |
|------|------|
| `filename` | CSV文件名 |
| `origin_code` | 起点机场代码（IATA） |
| `dest_code` | 终点机场代码（IATA） |
| `aircraft` | 飞机型号 |
| `wind_speed_ms` | 风速（m/s） |
| `aero_perturbation` | 气动扰动强度 |
| `noise_type` | 噪声类型 |
| `imu_noise_intensity` | IMU噪声强度（0-1） |
| `gps_noise_intensity` | GPS噪声强度（0-1） |

**行数**: 5,032行（包含表头）  
**用途**: 快速查询和筛选数据集

---

## 数据生成流程

### 1. 航线选择

- **来源**: `src/flightsim/data/waypoints.csv`
- **数量**: 处理前300条航线
- **覆盖范围**: 国内航线、区域航线、洲际航线

### 2. 噪声配置

每条航迹使用随机生成的噪声配置：

#### 过程噪声（Process Noise）
- **风场湍流**: 1-15 m/s（对应轻度到重度湍流）
- **气动扰动**: 0-0.2（无量纲）

#### 测量噪声（Measurement Noise）
- **基础强度**: 0.1-0.8（归一化系数）
- **特定参数**（根据噪声类型）:
  - 闪烁噪声: 概率 0.01-0.2, 幅度倍数 2-10
  - 漂移噪声: 漂移率 0.001-0.01
  - 有色噪声: 相关系数 0.6-0.95
  - 时变噪声: 周期 50-500, 幅度 0.5-3.0

### 3. 仿真执行

- **物理引擎**: 6-DOF（六自由度）动力学模型
- **自动驾驶**: 多阶段路径跟踪控制器
- **时间步长**: 0.1秒
- **终止条件**: 成功落地 或 达到最大时长限制（1.5倍预计飞行时间）

### 4. 质量验证

每条航迹经过以下检查：

| 验证项 | 标准 | 说明 |
|--------|------|------|
| **闭合性** | 最终高度 < 50m 且阶段为 ROLLOUT/TOUCHDOWN | 确保成功落地 |
| **下降时间** | < 3000秒（50分钟） | 检测异常下降 |

**通过标准**: 所有验证项都通过  
**自动标记**: `validation_passed` 字段记录结果

---

## 质量分析

### 数据质量总结

- **总生成数**: 5,444 条航迹
- **通过验证**: 5,031 条 (92.4%)
- **未通过验证**: 413 条 (7.6%)

### 失败案例分析

未通过验证的413条记录主要集中在以下两类边界情况：

#### 1. 极短途航线（~30%失败案例）

**典型航线**:
- PEK-TSN（北京-天津，~120km）
- SZX-CAN（深圳-广州，~100km）
- CAN-HKG（广州-香港，~130km）

**失败特征**:
- 最终高度: 65-875米
- 最终阶段: FINAL/FLARE
- **原因**: 自动驾驶逻辑未充分优化短距离进近程序

#### 2. 极长途洲际航线（~70%失败案例）

**典型航线**:
- PEK-JFK（北京-纽约，~11,000km）
- PVG-JFK（上海-纽约，~12,000km）
- CTU-LAX（成都-洛杉矶，~10,000km）
- SIN-JFK（新加坡-纽约，~15,000km）

**失败特征**:
- 最终高度: 13,000米左右
- 最终阶段: CRUISE（巡航）
- **原因**: 仿真时长限制（MAX_DURATION_MULTIPLIER=1.5），超长航线未能在限定时间内完成全程

#### 3. 下降时间异常（27条记录）

部分航线出现下降时间超过1000分钟（16小时）的情况，通常与航迹未闭合同时发生。

### 数据分布质量

✅ **中短途航线**（200-5000km）: >95% 成功率  
✅ **常规干线航线**: >98% 成功率  
⚠️ **极短途航线**（<200km）: ~60% 成功率  
⚠️ **极长途航线**（>10000km）: ~0% 成功率（时长限制）

---

## 使用说明

### 1. 访问数据集

```python
import pandas as pd
from pathlib import Path

# 通过索引文件加载
dataset_root = Path.home() / "share2/dataset/flightsim/v1"
index = pd.read_csv(dataset_root / "total_list.csv")

# 查看数据集信息
print(f"Total samples: {len(index)}")
print(f"\nNoise types:\n{index['noise_type'].value_counts()}")
print(f"\nAircraft types:\n{index['aircraft'].value_counts()}")
```

### 2. 加载特定航迹

```python
# 筛选白噪声 + A320 的航迹
filtered = index[(index['noise_type'] == 'white') & 
                 (index['aircraft'] == 'A320-200')]

# 加载第一条航迹数据
sample = filtered.iloc[0]
filepath = dataset_root / sample['noise_type'] / sample['filename']
trajectory = pd.read_csv(filepath)

print(f"Route: {sample['origin_code']}-{sample['dest_code']}")
print(f"Duration: {len(trajectory) * 0.1:.1f} seconds")
print(f"Trajectory shape: {trajectory.shape}")
```

### 3. 批量数据加载示例

```python
def load_trajectories(index_df, noise_type, max_samples=None):
    """批量加载指定噪声类型的航迹数据"""
    filtered = index_df[index_df['noise_type'] == noise_type]
    if max_samples:
        filtered = filtered.head(max_samples)
    
    trajectories = []
    for _, row in filtered.iterrows():
        filepath = dataset_root / row['noise_type'] / row['filename']
        traj = pd.read_csv(filepath)
        trajectories.append({
            'data': traj,
            'metadata': row.to_dict()
        })
    
    return trajectories

# 加载100条白噪声航迹
white_noise_data = load_trajectories(index, 'white', max_samples=100)
```

### 4. 数据集划分建议

```python
from sklearn.model_selection import train_test_split

# 按航线分层划分，避免数据泄露
routes = index['origin_code'] + '-' + index['dest_code']
index['route'] = routes

# 70% 训练集, 15% 验证集, 15% 测试集
train_idx, temp_idx = train_test_split(
    index.index, test_size=0.3, stratify=index['route'], random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=index.loc[temp_idx, 'route'], random_state=42
)

train_set = index.loc[train_idx]
val_set = index.loc[val_idx]
test_set = index.loc[test_idx]

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
```

---

## 重新生成数据集

如果需要重新生成 `total_list.csv` 索引文件（例如新增数据后），可以使用以下脚本：

```bash
cd /home/b220/share/user/wld/project/FlightSim/tools
python rebuild_total_list.py
```

该脚本会：
1. 遍历所有噪声类型目录
2. 读取所有JSON元数据文件
3. 仅提取通过验证的数据（`validation_passed == true`）
4. 生成新的 `total_list.csv` 文件

---

## 数据维护

### 验证日志位置
```
~/share2/dataset/flightsim/v1/validation_log.txt
```

查看失败案例：
```bash
grep '\[FAIL\]' ~/share2/dataset/flightsim/v1/validation_log.txt
```

### 存储空间
- **完整数据集**: ~13 GB
- **单个噪声类型**: ~2-3 GB
- **建议预留空间**: 20 GB

---

## 已知限制

1. **极短途航线**（<200km）成功率较低，自动驾驶系统需要优化短距离进近逻辑
2. **极长途洲际航线**（>10,000km）因仿真时长限制无法完成全程
3. **时变噪声**类型数据量相对较少（12.6%），是因为在生成过程中该类型失败率较高
4. **支线小飞机**（ERJ-145, CRJ-900）在某些短途航线上容易出现落地失败

---

## 引用说明

如果使用本数据集，请引用：

```
FlightSim Trajectory Dataset v1
Generated using FlightSim 6-DOF Flight Simulator
Date: 2026-01-08
```

---

## 更新历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1 | 2026-01-08 | 初始版本，包含5,031条航迹，覆盖279条航线，5种噪声类型 |

---

## 联系方式

如有问题或建议，请查看项目主 README 或提交 Issue。
