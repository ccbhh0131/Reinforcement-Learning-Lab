from collections import deque
from pathlib import Path

import numpy as np

from catch_env import CatchBallEnv
from agent import A2CAgent


# ============================================================
# PyCharm 运行设置
#
# 训练时：MODE = "train"
# 测试时：MODE = "test"
#
# 修改设置后，点击 PyCharm 右上角绿色 Run 按钮即可运行。
# ============================================================

MODE = "train"

# 训练局数
TRAIN_EPISODES = 1000

# 测试局数
TEST_EPISODES = 10

# 模型保存位置
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "catch_a2c_qcritic.pt"
)

# 训练时画面延迟：
# 0.0 表示尽量快速训练。
# 如果想观察训练画面，可以改成 0.01，但训练会变慢。
TRAIN_RENDER_DELAY = 0.0

# 测试时画面延迟，设置稍大一些便于观察
TEST_RENDER_DELAY = 0.03

# 每隔多少局打印一次训练信息
LOG_INTERVAL = 20

# 每隔多少局保存一次模型
SAVE_INTERVAL = 100


def train():
    """
    训练 Q-Critic 版 A2C 智能体。
    """
    print("开始训练 Q-Critic 版 A2C 小车接球智能体")
    print(f"训练局数：{TRAIN_EPISODES}")
    print(f"模型保存位置：{MODEL_PATH}")

    env = CatchBallEnv(
        render_delay=TRAIN_RENDER_DELAY
    )

    agent = A2CAgent(
        state_dim=5,
        action_dim=3,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        critic_coef=0.5
    )

    # 保存最近 100 局的奖励和接球结果
    recent_rewards = deque(maxlen=100)
    recent_catches = deque(maxlen=100)

    try:
        for episode in range(1, TRAIN_EPISODES + 1):
            # 开始新的一局
            state = env.reset()

            episode_reward = 0.0
            caught = False
            done = False

            while not done:
                # 训练时按 Actor 输出的概率抽样动作
                action = agent.select_action(
                    state,
                    deterministic=False
                )

                # 执行动作，获取下一状态、奖励和结束标志
                next_state, reward, done, info = env.step(action)

                # 使用这一条经验更新 Actor 和 Q-Critic
                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )

                # 更新当前状态和本局累计奖励
                state = next_state
                episode_reward += reward

                if done:
                    caught = bool(info.get("caught", False))

            # 记录本局结果
            recent_rewards.append(episode_reward)
            recent_catches.append(float(caught))

            # 定期打印最近 100 局的训练情况
            if episode % LOG_INTERVAL == 0:
                average_reward = float(np.mean(recent_rewards))
                catch_rate = float(np.mean(recent_catches))

                print(
                    f"Episode {episode:4d}/{TRAIN_EPISODES} | "
                    f"最近100局平均奖励：{average_reward:7.2f} | "
                    f"最近100局接球率：{catch_rate:.1%}"
                )

            # 定期保存模型
            if episode % SAVE_INTERVAL == 0:
                agent.save(str(MODEL_PATH))
                print(f"模型已保存：{MODEL_PATH}")

        # 训练结束后再保存一次
        agent.save(str(MODEL_PATH))
        print("训练完成。")
        print(f"最终模型已保存：{MODEL_PATH}")

    finally:
        # 无论训练正常结束还是发生异常，都尝试关闭环境窗口
        env.close()


def test():
    """
    加载训练好的模型并进行测试。
    """
    if not MODEL_PATH.exists():
        print(f"没有找到模型文件：{MODEL_PATH}")
        print("请先将 MODE 设置为 'train'，运行训练后再测试。")
        return

    print("开始测试 Q-Critic 版 A2C 模型")
    print(f"加载模型：{MODEL_PATH}")
    print(f"测试局数：{TEST_EPISODES}")

    env = CatchBallEnv(
        render_delay=TEST_RENDER_DELAY
    )

    agent = A2CAgent(
        state_dim=5,
        action_dim=3,
        hidden_dim=128
    )

    agent.load(str(MODEL_PATH))

    rewards = []
    catches = 0

    try:
        for episode in range(1, TEST_EPISODES + 1):
            state = env.reset()

            episode_reward = 0.0
            caught = False
            done = False

            while not done:
                # 测试时选择 Actor 概率最大的动作
                action = agent.select_action(
                    state,
                    deterministic=True
                )

                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    caught = bool(info.get("caught", False))

            rewards.append(episode_reward)
            catches += int(caught)

            result_text = "接住了" if caught else "漏接了"

            print(
                f"测试局 {episode:2d}/{TEST_EPISODES} | "
                f"奖励：{episode_reward:7.2f} | "
                f"结果：{result_text}"
            )

    finally:
        env.close()

    average_reward = float(np.mean(rewards))
    catch_rate = catches / TEST_EPISODES

    print("-" * 60)
    print(f"测试平均奖励：{average_reward:.2f}")
    print(f"测试接球率：{catch_rate:.1%}")


def main():
    """
    根据 MODE 选择训练或测试。
    """
    if MODE == "train":
        train()
    elif MODE == "test":
        test()
    else:
        print("MODE 设置错误，请设置为 'train' 或 'test'。")


if __name__ == "__main__":
    main()
