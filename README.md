# Fusion Congestion Control Algorithm

## 项目概述

Fusion是一个用于数据中心网络(DCN)的拥塞控制算法，采用数据-模型双驱动策略，结合传统模型映射和深度强化学习(DRL)技术。该算法使用PPO (Proximal Policy Optimization)动态优化拥塞控制参数，实现快速响应和优秀的网络适应性。

## 快速开始

### 安装

```bash
# 安装opengym模块 (Linux/Mac)
chmod +x install_opengym.sh
./install_opengym.sh

# Windows
install_opengym.bat

# 构建NS-3
./waf configure --enable-examples
./waf build
```

### 运行仿真

#### RL训练模式

```bash
# 启动NS-3仿真（启用Gym）
./waf --run "scratch/third mix/config_exp1_stable.txt"  # 修改配置文件中的ENABLE_GYM为1

# 运行RL训练（当前使用PPO算法）
python rl_train.py --port 5555
```

#### RL推理模式

```bash
# 启动NS-3仿真（启用Gym）
./waf --run "scratch/third mix/config_exp1_stable.txt"  # 修改配置文件中的ENABLE_GYM为1

# 运行RL推理
python rl_inference.py --model ./models/fusion_rl_model_final.pth --port 5555
```
