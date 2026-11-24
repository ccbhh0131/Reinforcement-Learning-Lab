"""在 VS Code 运行：python main.py all（完整实验）；python main.py quick（快速检查）。"""
import argparse
from datetime import datetime
from pathlib import Path
import torch
from lab import ROOT, DQN_BASE, A2C_BASE, comparison, sweep, evaluate, train, individual

def main():
    p = argparse.ArgumentParser(description="强化学习实验三、四：DQN 与 A2C")
    p.add_argument("mode", nargs="?", default="all",
                   choices=["all", "quick", "compare", "sweep", "dqn", "a2c", "demo", "teacher"])
    p.add_argument("--episodes", type=int, default=None, help="训练局数；默认对比1000、参数实验200")
    p.add_argument("--tests", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--render", action="store_true", help="训练时显示游戏窗口，会降低速度")
    p.add_argument("--model", type=Path, help="demo 模式的模型路径")
    p.add_argument("--algorithm", choices=["DQN", "A2C"], default="A2C")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if (a.episodes is not None and a.episodes < 1) or a.tests < 1 or a.window < 1:
        p.error("局数及窗口必须为正整数")
    if a.mode not in ("quick", "demo", "teacher") and a.tests < 20:
        p.error("正式实验测试至少20局")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = a.out or ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=True)
    print("结果目录：", root.resolve(), flush=True)
    if a.mode == "quick":
        # 冒烟测试：短训练、保存、重新加载、测试、绘图。不能用于正式报告。
        comparison(root / "quick", 5, 3, a.seed, 3, False, False)
    elif a.mode in ("teacher", "demo"):
        if a.mode == "teacher":
            model = ROOT / "teacher_models" / "catch_a2c_qcritic.pt"
            kind = "A2C"
        else:
            if a.model is None:
                p.error("demo 需要 --model 模型路径")
            model, kind = a.model, a.algorithm
        evaluate(kind, DQN_BASE if kind == "DQN" else A2C_BASE, model,
                 a.tests, a.seed+10000, root / "demo", render=True)
    elif a.mode in ("dqn", "a2c"):
        kind = a.mode.upper()
        config = DQN_BASE if kind == "DQN" else A2C_BASE
        directory = root / kind
        rows = train(kind, config, a.episodes or 1000, a.seed, directory, a.render)
        individual(rows, directory, kind, a.window)
        evaluate(kind, config, directory / "model.pt", a.tests, a.seed+10000, directory)
    else:
        if a.mode in ("all", "compare"):
            comparison(root / "comparison", a.episodes or 1000, a.tests, a.seed, a.window, a.render)
        if a.mode in ("all", "sweep"):
            sweep(root / "dqn_sweep", a.episodes or 200, a.tests, a.seed, a.window, a.render)
    print("\n完成！结果目录：", root.resolve(), flush=True)

if __name__ == "__main__":
    main()
