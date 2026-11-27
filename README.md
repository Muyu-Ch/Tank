# 坦克动荡（Pygame）

本地坦克迷宫游戏，支持单人对战、本地双人对战，以及挑战神经网络闪避 AI。炮弹会在墙壁上反弹。

## 启动

需要 Python 3.9 或更新版本：

```bash
python3 -m pip install -r requirements.txt
python3 tank.py
```

主菜单按 `1` 对战 AI 1.5，按 `2` 开始本地双人模式。单人模式每局生成随机迷宫和随机出生点。

| 操作 | 玩家 1 | 玩家 2 |
| --- | --- | --- |
| 前进 / 后退 | W / S | ↑ / ↓ |
| 左转 / 右转 | A / D | ← / → |
| 射击 | 空格 | 回车 |

游戏中按 `Esc` 暂停；暂停时按 `R` 重开比分，按 `M` 返回菜单。

## 当前 AI

[tank_ai.py](tank_ai.py) 中的 AI 1.5 使用规则规划器寻找直射或一次反弹的射线，调用真实炮弹物理验证后才开火。没有合适射线时，它会寻找可到达的射击位置，并在接近敌人时停止盲目逼近。闪避控制器使用 [tank_dodge_walls.py](tank_dodge_walls.py) 训练的两层、每层 64 个神经元的 DQN 网络，只接管坦克移动。密集炮火下，动作评分还会扣除后退撞墙和继续逼近墙角的动作。

网络读取坦克附近最多 24 面墙的相对矩形坐标，以及最近六发炮弹的位置与速度。训练场从四周射击，包含随机墙体和反弹射击；每存活一帧获得奖励。游戏中，控制器预测双方炮弹未来约 0.6 秒的轨迹，自己的炮弹在发射 0.42 秒后也视为危险。如果多发炮弹在未来聚集并逼近，AI 会提前进入闪避，优先选择远离弹群的位置，同时避免后退撞墙。进入闪避后至少保持约 0.4 秒，危险连续消失约 0.13 秒才恢复寻敌。

项目中只保留玩家模式实际使用的 `tank_dodge_walls_best.npz`。它已加入 `.gitignore`。

## 训练与评估

需要再次实验时，可从当前模型扩展成 96×96 网络训练 3000 局；候选结果默认写到系统临时目录，`--scratch` 可从零开始。评估和观看默认读取 AI 1.5 当前模型：

```bash
python3 tank_dodge_walls.py --episodes 3000 --seed 44
python3 tank_dodge_walls.py --evaluate 200 --seed 2026 --scenario crossfire
python3 tank_dodge_walls.py --evaluate 200 --seed 2026 --scenario corner
python3 tank_dodge_walls.py --watch
```
