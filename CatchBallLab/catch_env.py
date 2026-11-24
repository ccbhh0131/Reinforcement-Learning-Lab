import time
import tkinter as tk

import numpy as np


class CatchBallEnv:
    """
    可视化小车接球环境。

    状态 state：
        [
            小车位置,
            球的水平位置,
            球的高度,
            球的水平速度,
            预计落点与小车的相对距离
        ]

    动作 action：
        0：向左
        1：不动
        2：向右
    """

    def __init__(self, render_delay=0.005):
        self.render_delay = render_delay

        # 游戏窗口尺寸
        self.window_width = 640
        self.window_height = 400

        # 游戏中的位置用 [-1, 1] 表示
        self.paddle_x = 0.0
        self.ball_x = 0.0
        self.ball_y = 1.0
        self.ball_vx = 0.0

        self.done = False
        self.steps = 0

        # 游戏参数
        self.paddle_speed = 0.12
        self.paddle_half_width = 0.20
        self.ball_fall_speed = 1.0 / 25.0
        self.max_steps = 25

        self.closed = False

        # 创建 Tkinter 窗口
        self.root = tk.Tk()
        self.root.title("DQN 小车接球")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(
            self.root,
            width=self.window_width,
            height=self.window_height,
            bg="white"
        )
        self.canvas.pack()

        self.status_text = self.canvas.create_text(
            320,
            22,
            text="小车接球：接住球得分，漏接球扣分",
            font=("Arial", 15)
        )

        self.reset()

    def close(self):
        """关闭游戏窗口。"""
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _world_x_to_screen(self, x):
        """把 [-1, 1] 的横坐标转换成屏幕坐标。"""
        margin = 35
        usable_width = self.window_width - 2 * margin
        return margin + (x + 1.0) * 0.5 * usable_width

    def _world_y_to_screen(self, y):
        """把 [0, 1] 的球高度转换成屏幕坐标。"""
        top = 55
        bottom = 315
        return bottom - y * (bottom - top)

    def _predict_landing_x(self):
        """
        简单预测球落地时的水平位置。
        预测时考虑球碰到左右边界后的反弹。
        """
        remaining_steps = max(
            0,
            int(round(self.ball_y / self.ball_fall_speed))
        )

        x = self.ball_x
        vx = self.ball_vx

        for _ in range(remaining_steps):
            x += vx

            if x > 1.0:
                x = 2.0 - x
                vx = -abs(vx)
            elif x < -1.0:
                x = -2.0 - x
                vx = abs(vx)

        return float(np.clip(x, -1.0, 1.0))

    def _get_state(self):
        """生成神经网络使用的状态向量。"""
        landing_x = self._predict_landing_x()

        state = np.array(
            [
                self.paddle_x,
                self.ball_x,
                self.ball_y,
                self.ball_vx / 0.035,
                (landing_x - self.paddle_x) / 2.0
            ],
            dtype=np.float32
        )

        return state

    def reset(self):
        """开始新的一局，返回初始状态。"""
        if self.closed:
            raise RuntimeError("游戏窗口已经关闭。")

        self.paddle_x = 0.0

        # 随机生成球的位置、方向和速度
        self.ball_x = float(np.random.uniform(-0.85, 0.85))
        direction = int(np.random.choice([-1, 1]))
        speed = float(np.random.uniform(0.015, 0.035))
        self.ball_vx = direction * speed

        self.ball_y = 1.0
        self.steps = 0
        self.done = False

        self.render()

        return self._get_state()

    def step(self, action):
        """
        执行动作，返回：
            next_state, reward, done, info
        """
        if self.closed:
            raise RuntimeError("游戏窗口已经关闭。")

        if self.done:
            raise RuntimeError("本局已经结束，请先调用 reset()。")

        old_landing_x = self._predict_landing_x()
        old_distance = abs(old_landing_x - self.paddle_x)

        # 1. 根据动作移动小车
        if action == 0:
            self.paddle_x -= self.paddle_speed
        elif action == 1:
            pass
        elif action == 2:
            self.paddle_x += self.paddle_speed
        else:
            raise ValueError("action 必须是 0、1 或 2。")

        self.paddle_x = float(np.clip(self.paddle_x, -1.0, 1.0))

        # 2. 球水平移动；碰到边界后反弹
        self.ball_x += self.ball_vx

        if self.ball_x > 1.0:
            self.ball_x = 2.0 - self.ball_x
            self.ball_vx = -abs(self.ball_vx)
        elif self.ball_x < -1.0:
            self.ball_x = -2.0 - self.ball_x
            self.ball_vx = abs(self.ball_vx)

        # 3. 球向下落
        self.ball_y -= self.ball_fall_speed
        self.steps += 1

        # 4. 过程奖励：小车向预计落点靠近时，得到一点奖励
        new_landing_x = self._predict_landing_x()
        new_distance = abs(new_landing_x - self.paddle_x)

        reward = 0.1 * (old_distance - new_distance) - 0.01

        info = {
            "caught": False,
            "landing_x": new_landing_x
        }

        # 5. 球落到底部，判断是否接住
        if self.ball_y <= 0.0 or self.steps >= self.max_steps:
            self.ball_y = 0.0
            self.done = True

            caught = (
                abs(self.ball_x - self.paddle_x)
                <= self.paddle_half_width
            )

            if caught:
                reward += 5.0
            else:
                reward -= 5.0

            info["caught"] = caught

        next_state = self._get_state()

        self.render()

        return next_state, float(reward), self.done, info

    def render(self, episode=None, epsilon=None):
        """绘制当前游戏画面。"""
        if self.closed:
            return

        try:
            self.canvas.delete("game")

            # 左右边界
            left_x = self._world_x_to_screen(-1.0)
            right_x = self._world_x_to_screen(1.0)

            self.canvas.create_line(
                left_x, 45, left_x, 340,
                fill="gray",
                width=2,
                tags="game"
            )
            self.canvas.create_line(
                right_x, 45, right_x, 340,
                fill="gray",
                width=2,
                tags="game"
            )

            # 球的屏幕坐标
            ball_screen_x = self._world_x_to_screen(self.ball_x)
            ball_screen_y = self._world_y_to_screen(self.ball_y)

            # 绘制球
            radius = 12
            self.canvas.create_oval(
                ball_screen_x - radius,
                ball_screen_y - radius,
                ball_screen_x + radius,
                ball_screen_y + radius,
                fill="orange",
                outline="darkorange",
                width=2,
                tags="game"
            )

            # 绘制小车
            paddle_screen_x = self._world_x_to_screen(self.paddle_x)
            paddle_y = 330
            paddle_width = 110
            paddle_height = 20

            self.canvas.create_rectangle(
                paddle_screen_x - paddle_width / 2,
                paddle_y,
                paddle_screen_x + paddle_width / 2,
                paddle_y + paddle_height,
                fill="steelblue",
                outline="navy",
                width=2,
                tags="game"
            )

            # 显示当前信息
            message = (
                f"小车位置: {self.paddle_x:+.2f}    "
                f"球位置: {self.ball_x:+.2f}    "
                f"球高度: {self.ball_y:.2f}"
            )

            if episode is not None:
                message += f"    Episode: {episode}"

            if epsilon is not None:
                message += f"    ε: {epsilon:.2f}"

            self.canvas.itemconfigure(
                self.status_text,
                text=message
            )

            self.root.update_idletasks()
            self.root.update()

            if self.render_delay > 0:
                time.sleep(self.render_delay)

        except tk.TclError:
            self.closed = True
