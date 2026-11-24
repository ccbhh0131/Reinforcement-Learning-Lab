import os

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    共享特征网络 + Actor + Q-Critic。

    输入：
        state：5 维状态

    输出：
        action_probs：左、停、右三个动作的概率
        q_values：左、停、右三个动作对应的 Q 值
    """

    def __init__(self, state_dim=5, action_dim=3, hidden_dim=128):
        super().__init__()

        # Actor 和 Critic 共享的状态特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor 输出每个动作的概率
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Q-Critic 输出每个动作对应的 Q 值
        self.critic_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        参数：
            state: shape 为 [batch_size, state_dim] 的状态张量

        返回：
            action_probs: shape 为 [batch_size, action_dim]
            q_values: shape 为 [batch_size, action_dim]
        """
        features = self.shared(state)

        action_probs = torch.softmax(
            self.actor_head(features),
            dim=-1
        )

        q_values = self.critic_head(features)

        return action_probs, q_values


class A2CAgent:
    """
    使用 Q-Critic 的 Advantage Actor-Critic 智能体。

    环境动作编号：
        0：向左
        1：不动
        2：向右
    """

    def __init__(
        self,
        state_dim=5,
        action_dim=3,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        critic_coef=0.5,
        device=None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef

        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )

    def select_action(self, state, deterministic=False):
        """
        根据当前状态选择动作。

        deterministic=False：
            按 Actor 输出的概率抽样，训练时使用。

        deterministic=True：
            选择概率最大的动作，测试时使用。
        """
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            distribution = Categorical(probs=action_probs)
            action = distribution.sample()

        return int(action.item())

    def update(self, state, action, reward, next_state, done):
        """
        使用一条经验更新 Actor 和 Q-Critic。

        输入经验：
            state：当前状态
            action：执行的动作
            reward：环境返回的奖励
            next_state：执行动作后的状态
            done：本局是否结束
        """
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        next_state_tensor = torch.as_tensor(
            next_state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        action_tensor = torch.tensor(
            [action],
            dtype=torch.long,
            device=self.device
        )

        reward_tensor = torch.tensor(
            [reward],
            dtype=torch.float32,
            device=self.device
        )

        done_tensor = torch.tensor(
            [float(done)],
            dtype=torch.float32,
            device=self.device
        )

        # 当前状态下：
        # action_probs：策略概率 pi(a|s)
        # q_values：各动作的 Q(s,a)
        action_probs, q_values = self.network(state_tensor)

        # 取出实际执行动作对应的 Q(s_t, a_t)
        q_value = q_values.gather(
            dim=1,
            index=action_tensor.unsqueeze(1)
        ).squeeze(1)

        # 当前策略下的状态价值：
        # V(s) = sum_a pi(a|s) * Q(s,a)
        state_value = (
            action_probs * q_values
        ).sum(dim=1)

        # 计算下一状态的 TD target。
        # 下一状态不需要反向传播梯度。
        with torch.no_grad():
            next_action_probs, next_q_values = self.network(
                next_state_tensor
            )

            # 下一状态在当前策略下的期望 Q 值：
            # V(s') = sum_a' pi(a'|s') * Q(s',a')
            next_state_value = (
                next_action_probs * next_q_values
            ).sum(dim=1)

            # TD target：
            # y = r + gamma * (1-done) * V(s')
            target = (
                reward_tensor
                + self.gamma
                * (1.0 - done_tensor)
                * next_state_value
            )

        # Advantage：
        # A(s,a) = Q(s,a) - V(s)
        advantage = q_value - state_value

        # Critic 的 TD error：
        # delta = target - Q(s,a)
        td_error = target - q_value

        # Actor 根据 Advantage 调整动作概率
        distribution = Categorical(probs=action_probs)
        log_prob = distribution.log_prob(action_tensor)

        actor_loss = -(
            log_prob * advantage.detach()
        ).mean()

        # Critic 学习让 Q(s,a) 接近 TD target
        critic_loss = td_error.pow(2).mean()

        # 熵鼓励训练过程保留一定探索
        entropy = distribution.entropy().mean()

        # 总损失
        loss = (
            actor_loss
            + self.critic_coef * critic_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()

        # 限制梯度大小，帮助训练稳定
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "advantage": float(advantage.detach().mean().cpu()),
            "td_error": float(td_error.detach().mean().cpu())
        }

    def save(self, path):
        """
        保存网络参数、优化器状态和训练参数。
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "gamma": self.gamma,
                "entropy_coef": self.entropy_coef,
                "critic_coef": self.critic_coef
            },
            path
        )

    def load(self, path):
        """
        加载网络参数和优化器状态。
        """
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.network.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        # 加载保存的训练参数；如果旧模型中没有这些字段，则保留当前值
        self.gamma = checkpoint.get("gamma", self.gamma)
        self.entropy_coef = checkpoint.get(
            "entropy_coef",
            self.entropy_coef
        )
        self.critic_coef = checkpoint.get(
            "critic_coef",
            self.critic_coef
        )

        self.network.eval()
