"""统一实验驱动。教师原始环境与 A2C 实现保持不变。"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from catch_env import CatchBallEnv
from agent import A2CAgent
from dqn_agent import DQNAgent

ROOT = Path(__file__).resolve().parent
DQN_BASE = dict(lr=0.001, gamma=0.95, capacity=2000, batch_size=64, target_interval=10)
A2C_BASE = dict(learning_rate=3e-4, gamma=0.99, entropy_coef=0.01, critic_coef=0.5)
FIELDS = ["episode", "reward", "caught", "loss", "actor_loss", "critic_loss", "epsilon", "entropy_coef"]

# 只跳过 Tk 窗口初始化和渲染；reset、step、奖励、落点预测完全复用教师原版。
# 这些初始字段与教师 CatchBallEnv.__init__ 一致。
class FastEnv(CatchBallEnv):
    def __init__(self):
        self.render_delay = 0.0
        self.window_width, self.window_height = 640, 400
        self.paddle_x = self.ball_x = self.ball_vx = 0.0
        self.ball_y = 1.0
        self.done = False
        self.steps = 0
        self.paddle_speed = 0.12
        self.paddle_half_width = 0.20
        self.ball_fall_speed = 1.0 / 25.0
        self.max_steps = 25
        self.closed = False
        self.reset()
    def render(self, *args, **kwargs):
        pass
    def close(self):
        self.closed = True

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def new_agent(kind, config):
    if kind == "DQN":
        return DQNAgent(**config)
    return A2CAgent(device="cpu", **config)

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")

def train(kind, config, episodes, seed, directory, render=False, entropy_decay=False):
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    agent = new_agent(kind, config)
    env = CatchBallEnv(render_delay=0.0) if render else FastEnv()
    rows = []
    metadata = dict(algorithm=kind, config=config, episodes=episodes, seed=seed,
                    entropy_decay=entropy_decay, device="cpu", render=render,
                    epsilon_start=1.0 if kind == "DQN" else None,
                    epsilon_end=0.05 if kind == "DQN" else None)
    (directory / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n开始训练 {directory.name}: {episodes} 局, seed={seed}", flush=True)
    try:
        with (directory / "train.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for ep in range(1, episodes + 1):
                fraction = (ep - 1) / max(1, episodes - 1)
                epsilon = 1.0 - 0.95 * fraction if kind == "DQN" else None
                if kind != "DQN" and entropy_decay:
                    agent.entropy_coef = config["entropy_coef"] * (1 - 0.9 * fraction)
                state = env.reset()
                done, total, caught = False, 0.0, False
                losses, actor_losses, critic_losses = [], [], []
                while not done:
                    if kind == "DQN":
                        action = agent.select_action(state, epsilon)
                    else:
                        action = agent.select_action(state, deterministic=False)
                    ns, reward, done, info = env.step(action)
                    if kind == "DQN":
                        agent.remember(state, action, reward, ns, done)
                        loss = agent.update()
                        if loss is not None:
                            losses.append(loss)
                    else:
                        metrics = agent.update(state, action, reward, ns, done)
                        actor_losses.append(metrics["actor_loss"])
                        critic_losses.append(metrics["critic_loss"])
                    total += reward
                    state = ns
                    caught = bool(info["caught"])
                if kind == "DQN" and ep % agent.target_interval == 0:
                    agent.sync_target()
                row = dict(episode=ep, reward=total, caught=int(caught),
                           loss=mean_or_nan(losses), actor_loss=mean_or_nan(actor_losses),
                           critic_loss=mean_or_nan(critic_losses), epsilon=epsilon,
                           entropy_coef=agent.entropy_coef if kind != "DQN" else None)
                rows.append(row)
                writer.writerow(row)
                f.flush()
                if ep == 1 or ep % 20 == 0 or ep == episodes:
                    tail = rows[-100:]
                    print(f"{directory.name} {ep:4d}/{episodes} | 近{len(tail)}局平均奖励 {np.mean([x['reward'] for x in tail]):.3f} | 接球率 {np.mean([x['caught'] for x in tail]):.1%}", flush=True)
                if ep % 100 == 0 or ep == episodes:
                    agent.save(str(directory / "model.pt"))
    finally:
        env.close()
    return rows

def evaluate(kind, config, model, episodes, seed, directory, render=False):
    agent = new_agent(kind, config)
    agent.load(str(model))
    seed_all(seed)
    env = CatchBallEnv(render_delay=0.03) if render else FastEnv()
    rows = []
    try:
        for ep in range(1, episodes + 1):
            state = env.reset()
            done, total = False, 0.0
            while not done:
                action = agent.select_action(state, 0.0) if kind == "DQN" else agent.select_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                total += reward
            rows.append(dict(episode=ep, reward=total, caught=int(info["caught"])))
            if render:
                print(f"测试 {ep}/{episodes} | 奖励 {total:.3f} | {'接住' if info['caught'] else '漏接'}", flush=True)
    finally:
        env.close()
    write_csv(directory / "test.csv", rows, ["episode", "reward", "caught"])
    result = dict(test_episodes=episodes, test_seed=seed,
                  average_reward=float(np.mean([r["reward"] for r in rows])),
                  catch_rate=float(np.mean([r["caught"] for r in rows])),
                  catches=sum(r["caught"] for r in rows))
    print(f"测试结果 {directory.name}: 平均奖励 {result['average_reward']:.3f}，接球率 {result['catch_rate']:.1%}", flush=True)
    return result

def rolling(values, window):
    return np.array([np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))])

def configure_fonts():
    names = {x.name for x in font_manager.fontManager.ttflist}
    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        if font in names:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False

ZH = configure_fonts()
def label(zh, en):
    return zh if ZH else en

def curves(runs, metric, window, path, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, rows in runs.items():
        y = [r[metric] for r in rows]
        if metric == "caught":
            y = np.array(y) * 100
        ax.plot(range(1, len(y)+1), rolling(y, window), label=name, linewidth=1.6)
    ax.set_xlabel(label("训练局数", "Episode"))
    ax.set_ylabel(label("平均奖励", "Mean reward") if metric == "reward" else label("接球率 (%)", "Catch rate (%)"))
    ax.set_title(f"{title} (window={window})")
    if metric == "caught":
        ax.set_ylim(0, 102)
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

def individual(rows, directory, kind, window):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = range(1, len(rows)+1)
    rewards = [r["reward"] for r in rows]
    axes[0].plot(x, rewards, alpha=.3, label="Reward")
    axes[0].plot(x, rolling(rewards, window), label=f"Mean ({window})")
    axes[0].set_ylabel(label("奖励", "Reward"))
    axes[1].plot(x, rolling([r["caught"]*100 for r in rows], window), label=f"Catch rate ({window})")
    axes[1].set_ylabel(label("接球率 (%)", "Catch rate (%)"))
    axes[1].set_ylim(0, 102)
    axes[1].set_xlabel(label("训练局数", "Episode"))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle(kind)
    fig.tight_layout()
    fig.savefig(directory / "training_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = ["loss"] if kind == "DQN" else ["actor_loss", "critic_loss"]
    for key in keys:
        ax.plot(x, [r[key] for r in rows], label=key, alpha=.8)
    ax.set_xlabel(label("训练局数", "Episode"))
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(directory / "loss_curve.png", dpi=160)
    plt.close(fig)

def comparison(root, episodes, tests, seed, window, render=False, optimize=True):
    runs, summary = {}, []
    specs = [("DQN", "DQN", DQN_BASE, False), ("A2C", "A2C", A2C_BASE, False)]
    if optimize:
        specs += [("A2C_entropy_decay", "A2C", A2C_BASE, True),
                  ("A2C_lr_0.0001", "A2C", dict(A2C_BASE, learning_rate=1e-4), False)]
    for name, kind, config, decay in specs:
        directory = root / name
        rows = train(kind, config, episodes, seed, directory, render, decay)
        runs[name] = rows
        individual(rows, directory, kind, window)
        result = evaluate(kind, config, directory / "model.pt", tests, seed+10000, directory)
        summary.append(dict(algorithm=name, train_episodes=episodes, **result))
    write_csv(root / "summary.csv", summary, list(summary[0]))
    base = {name: runs[name] for name in ("DQN", "A2C")}
    for metric in ("reward", "caught"):
        curves(base, metric, window, root / f"compare_{metric}.png", "DQN vs A2C")
        if optimize:
            variants = {name: rows for name, rows in runs.items() if name != "DQN"}
            curves(variants, metric, window, root / f"optimize_{metric}.png", "A2C optimization")
            curves(runs, metric, window, root / f"all_{metric}.png", "All methods")
    return summary

def sweep(root, episodes, tests, seed, window, render=False):
    # 实验三第4、5页：控制变量，第四组按 Batch Size 处理。
    groups = dict(capacity=[1000, 2000, 4000, 8000],
                  target_interval=[5, 10, 15, 20],
                  lr=[0.0001, 0.001, 0.005, 0.01],
                  batch_size=[32, 64, 128])
    base_dir = root / "baseline"
    base_rows = train("DQN", DQN_BASE, episodes, seed, base_dir, render)
    individual(base_rows, base_dir, "DQN", window)
    base_test = evaluate("DQN", DQN_BASE, base_dir / "model.pt", tests, seed+10000, base_dir)
    summary = []
    for key, values in groups.items():
        runs = {}
        for value in values:
            name = f"{key}={value}"
            if value == DQN_BASE[key]:
                rows, result = base_rows, base_test
            else:
                directory = root / key / str(value)
                config = dict(DQN_BASE, **{key: value})
                rows = train("DQN", config, episodes, seed, directory, render)
                result = evaluate("DQN", config, directory / "model.pt", tests, seed+10000, directory)
            runs[name] = rows
            tail = rows[-min(50, episodes):]
            summary.append(dict(parameter=key, value=value, train_episodes=episodes,
                                tail_mean_reward=float(np.mean([x["reward"] for x in tail])),
                                tail_catch_rate=float(np.mean([x["caught"] for x in tail])),
                                **result))
        curves(runs, "reward", window, root / f"{key}_reward.png", f"DQN {key}")
        curves(runs, "caught", window, root / f"{key}_catch.png", f"DQN {key}")
    write_csv(root / "summary.csv", summary, list(summary[0]))
    return summary
