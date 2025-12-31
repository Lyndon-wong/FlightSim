---
title: FlightSim
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.12.0
app_file: examples/gradio_demo.py
pinned: false
license: mit
---

# FlightSim ✈️

FlightSim 是一个独立的、高性能的民航飞行轨迹生成与仿真库。基于六自由度（6-DOF）动力学模型，能够生成高保真的飞行轨迹数据。

## 功能特性

*   **高保真仿真**: 基于物理的 6-DOF 动力学模型，考虑气动力、推力、重力和大气环境。
*   **完整的飞行阶段**: 模拟从滑行、起飞、爬升、巡航、下降、进近到着陆的完整飞行过程（共17个阶段）。
*   **丰富的机型库**: 内置 20 种常见民航客机参数（如 A320, B737, B777 等），支持窄体、宽体及支线客机。
*   **全球航线支持**: 内置 300 条全球代表性航线数据。
*   **易于使用**: 提供简洁的 Python API 和 Gradio 可视化演示界面。

## 目录结构

```
FlightSim/
├── src/flightsim/       # 核心源代码包
├── data/                # 数据文件 (航线、机型参数)
├── examples/            # 示例脚本和 Notebooks
├── tools/               # 辅助工具脚本
├── tests/               # 测试代码
└── pyproject.toml       # 项目配置文件
```

## 快速开始

### 安装

建议在虚拟环境中安装：

```bash
cd FlightSim
pip install -e .
```

### 运行示例

生成简单的飞行轨迹：

```bash
python examples/simple_trajectory.py
```

这将生成 CSV 格式的轨迹文件。

### 运行可视化 Demo

启动 Gradio 交互式界面：

```bash
python examples/gradio_demo.py
```

在浏览器中打开显示的 URL 即可使用。

### 部署到 Hugging Face Spaces

本项目已配置为可以直接部署到 [Hugging Face Spaces](https://huggingface.co/spaces)。

1.  在 Hugging Face 创建一个新的 Space。
2.  选择 **Gradio** 作为 SDK。
3.  将本项目的所有文件上传到 Space 的仓库中。
4.  Space 将自动检测 `README.md` 中的配置并启动演示。

## 核心模块

*   `flightsim.sixdof`: 六自由度运动方程解算。
*   `flightsim.autopilot`: 自动驾驶仪逻辑，管理飞行阶段和控制律。
*   `flightsim.aerodynamics`: 气动参数管理与计算。
*   `flightsim.navigation`: 导航计算工具。

## 数据说明

*   `data/waypoints.csv`: 包含 300 条真实航线数据。
*   `data/plans.csv`: 包含 20 种飞机的性能和气动参数。

## 许可证

MIT License
