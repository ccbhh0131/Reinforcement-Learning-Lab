"""实验三：标准 DQN（经验回放、目标网络、epsilon-greedy）。"""
import random
from collections import deque
import numpy as np
import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 3))
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.95, capacity=2000, batch_size=64,
                 target_interval=10):
        if capacity < batch_size:
            raise ValueError("经验池容量不能小于 batch size")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_interval = target_interval
        self.online = QNetwork()
        self.target = QNetwork()
        self.sync_target()
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = deque(maxlen=capacity)

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(3)
        with torch.no_grad():
            return int(self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, ns, done = zip(*random.sample(self.memory, self.batch_size))
        s = torch.as_tensor(np.stack(s), dtype=torch.float32)
        ns = torch.as_tensor(np.stack(ns), dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.long)
        r = torch.as_tensor(r, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32)
        q = self.online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1 - done) * self.target(ns).max(1).values
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save({"model_state_dict": self.online.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.online.load_state_dict(checkpoint["model_state_dict"])
        self.sync_target()
        self.online.eval()
