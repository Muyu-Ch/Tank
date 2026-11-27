# 坦克动荡（Pygame）

本地坦克迷宫游戏，支持单人对战、本地双人对战，以及挑战神经网络闪避 AI。炮弹会在墙壁上反弹。

## 启动

需要 Python 3.9 或更新版本：

```bash
python3 -m pip install -r requirements.txt
python3 tank.py
```

主菜单按 `1` 开始普通单人模式，按 `2` 开始本地双人模式，按 `3` 挑战闪避 AI。第 3 项每局生成随机迷宫和随机出生点。

| 操作 | 玩家 1 | 玩家 2 |
| --- | --- | --- |
| 前进 / 后退 | W / S | ↑ / ↓ |
| 左转 / 右转 | A / D | ← / → |
| 射击 | 空格 | 回车 |

游戏中按 `Esc` 暂停；暂停时按 `R` 重开比分，按 `M` 返回菜单。

## 当前 AI

[tank_ai.py](tank_ai.py) 中的 AI 1.5 使用规则规划器寻找直射或一次反弹的射线，调用真实炮弹物理验证后才开火。没有合适射线时，它会寻找可到达的射击位置，并在接近敌人时停止盲目逼近。闪避控制器使用 [tank_dodge_walls.py](tank_dodge_walls.py) 训练的两层、每层 64 个神经元的 DQN 网络，只接管坦克移动。密集炮火下，动作评分还会扣除后退撞墙和继续逼近墙角的动作。

网络读取坦克附近最多 24 面墙的相对矩形坐标，以及最近六发炮弹的位置与速度。训练场从四周射击，包含随机墙体和反弹射击；每存活一帧获得奖励。游戏中，控制器预测双方炮弹未来约 0.6 秒的轨迹，自己的炮弹在发射 0.42 秒后也视为危险。如果多发炮弹在未来聚集并逼近，AI 会提前进入闪避，优先选择远离弹群的位置，同时避免后退撞墙。进入闪避后至少保持约 0.4 秒，危险连续消失约 0.13 秒才恢复寻敌。

玩家模式当前使用 `tank_dodge_walls_best.npz`。新训练候选模型单独保存在 `tank_dodge_dense_best.npz`；它采用 96×96 神经元、3000 局训练，并加入交叉火力和近墙出生。独立评估中，新模型的贴墙时间有所下降，但存活时间略低，因此尚未替换玩家模式的模型。两个模型文件均已加入 `.gitignore`。

## 训练与评估

默认从旧模型扩展成 96×96 网络训练 3000 局；`--scratch` 可从零开始。继续训练使用 `--resume`：

```bash
python3 tank_dodge_walls.py --episodes 3000 --seed 44 --validation-games 40
python3 tank_dodge_walls.py --resume --episodes 3000 --seed 45 --validation-games 40
python3 tank_dodge_walls.py --evaluate 200 --seed 2026 --scenario crossfire
python3 tank_dodge_walls.py --evaluate 200 --seed 2026 --scenario corner
python3 tank_dodge_walls.py --watch
```

[tank_duel.py](tank_duel.py) 在随机迷宫中对战规则 AI。默认使用玩家模式的 AI 1.5；`--ai13` 使用旧版 AI 1.3；`--legacy` 使用更早的即时切换控制器；`--planner-only` 只使用旧寻敌规则。每种配置评估时使用相同的种子、地图和出生点。

```bash
python3 tank_duel.py --games 100 --seed 2026
python3 tank_duel.py --games 100 --seed 2026 --model tank_dodge_dense_best.npz
python3 tank_duel.py --games 100 --seed 2026 --ai13
python3 tank_duel.py --games 100 --seed 2026 --legacy
python3 tank_duel.py --games 100 --seed 2026 --planner-only
python3 tank_duel.py --watch
```

`--watch` 中按 `P` 暂停、`N` 下一局、`Esc` 退出。评估输出包含胜负、自弹死亡、反弹击杀、自弹和弹群威胁触发次数、闪避接管次数和控制器切换次数。
