# 强化学习实验三、四：DQN 与 A2C

## Windows / VS Code 使用

下载仓库 ZIP 并解压，使用 VS Code 打开 **CatchBallLab** 文件夹（不是仓库外层）。
Python 3.9+，建议已安装的 Python 3.12。依赖为 numpy、matplotlib、torch；Tkinter 用于游戏展示。

打开终端（Ctrl+J），检查依赖：

    python -c "import torch,numpy,matplotlib,tkinter; print('OK')"

先做快速检查（每种算法5局训练、3局测试；仅用于检查代码）：

    python main.py quick

正式实验四（1000局训练、100局测试，含两组 A2C 单变量优化）：

    python main.py compare

实验三四组 DQN 超参数对比（每组200局）：

    python main.py sweep

也可一条命令运行两个实验：

    python main.py all

不带参数运行 main.py 同样执行 all。训练默认无画面，速度更快；想观察实时训练可加 --render。训练结束会自动生成真实 CSV、PNG 和模型，不预填数据。

## 显示游戏与测试

测试教师提供的原始 Q-Critic 模型（20局，不覆盖原权重）：

    python main.py teacher --tests 20

测试自己训练的模型（把路径换成终端输出的实际目录，路径有空格要加引号）：

    python main.py demo --algorithm A2C --model "results/实际时间/comparison/A2C/model.pt" --tests 20

DQN 模型将 --algorithm 改为 DQN，并填写 DQN/model.pt 路径。
单独训练也可用 python main.py dqn 或 python main.py a2c。

## 文件说明

- catch_env.py：教师原版，实验三、四完全相同；未修改。
- agent.py：教师原版 A2C，未修改。
- dqn_agent.py：标准 DQN 实现，包含在线/目标网络、经验回放、epsilon-greedy、TD 更新。
- lab.py：统一训练/测试、CSV、可视化、超参数实验。
- main.py：一键入口。
- teacher_models：教师原始 Q-Critic 权重，独立保存。
- results：运行后自动生成，按时间隔离，避免覆盖上次结果。

FastEnv 继承教师环境，复用 reset、step 和全部奖励逻辑；只跳过 Tk 初始化与渲染。训练使用 CPU；测试可开启原版 GUI。所有正式对比均从随机初始化开始，教师权重只用于 demo。
实验三 DQN 基准：gamma=0.95，lr=0.001，经验池2000，batch64，每10局同步目标网络，epsilon 从1线性降到0.05。四组比较经验池容量、同步间隔、学习率、Batch Size（指导书第4页最后一行疑似笔误，按第5页说明执行）。
实验四 A2C 基准：教师 Q-Critic、gamma=0.99、lr=0.0003、entropy=0.01、critic_coef=0.5。额外两组单变量实验：熵系数从0.01线性降至0.001；学习率改为0.0001。不会预设优化一定提升，报告据实际结果分析。
默认训练 seed=42，测试 seed=10042；各算法同组使用相同种子和测试局数，测试采用确定性动作，不更新参数。可使用 --seed 43、--seed 44 重复验证，避免单个种子下的偶然结论。滑动窗口默认50局，前不足50局按已有数据平均。

## 报告需要的文件

- comparison/summary.csv：DQN、A2C 及两个优化方案测试奖励和接球率（0~1）。
- comparison/compare_reward.png、compare_caught.png：原版 DQN 与 A2C 同图对比。
- comparison/optimize_reward.png、optimize_caught.png：A2C 单变量优化对比。
- comparison/A2C/training_curve.png：A2C 每局奖励、滑动平均奖励及接球率。
- comparison/各算法/train.csv、test.csv、config.json、loss_curve.png、model.pt。
- dqn_sweep/summary.csv：各参数末50局表现与独立测试结果。
- dqn_sweep/*_reward.png、*_catch.png：实验三控制变量对比图。
- dqn_sweep/baseline/training_curve.png、loss_curve.png：实验三基准奖励及 Loss。
- 需要截图时使用 teacher 或 demo 模式，截图真实窗口及终端输出。
