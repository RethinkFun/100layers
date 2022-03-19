import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from replaybuffer import ValueReplayBuffer
import numpy as np
from nsshaft import NSSHAFT
import time


class RLModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.back_bone = tf.keras.applications.ResNet50V2(include_top=False, weights=None)
        self.valuenet = tf.keras.Sequential(
            [Flatten(),
             Dense(32, activation='relu'),
             Dense(64, activation='relu'),
             Dense(5)]
        )

    def call(self, inputs, training=True):
        inputs = inputs / 255
        y = self.back_bone(inputs, training=training)
        y = self.valuenet(y, training=training)
        return y


class Agent:
    def __init__(self, env):
        self.env = env
        self.action_dim = self.env.action_dim()

        def create_model():
            model = RLModel()
            return model

        self.model = create_model()
        #self.model = tf.keras.models.load_model("model120")
        self.train_episode = 20000
        self.exploration_episode = 200
        self.epsilon_max = 0.3
        self.epsilon_min = 0.0
        self.batch_size = 32
        self.gamma = 1
        self.buffer = ValueReplayBuffer(30000)
        self.optimizer = tf.keras.optimizers.Adam(0.0001)
        self.train_num = 0
        self.train_loss = tf.metrics.Mean()
        self.action_loss = tf.metrics.Mean()
        self.pos_loss = tf.metrics.Mean()
        self.layers_mean = tf.metrics.Mean()

    def choose_action(self, state, episode, x):
        epsilon = self.epsilon_max - min(1, episode / self.exploration_episode) * (self.epsilon_max - self.epsilon_min)
        if np.random.uniform() < epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_value = self.model(state[np.newaxis, :])[0]
            action = np.argmax(q_value[:3])
        return action

    def test_episode(self, test_episodes):
        for episode in range(test_episodes):
            state, x, y = self.env.reset()
            total_reward, done = 0, False
            while not done:
                action = self.model(state[np.newaxis, :], False)[0][:3]
                action = np.argmax(action)
                next_state, reward, done, f, x, y, l = self.env.step(action)
                total_reward += reward
                state = next_state
            print('Test episode:{}, layers:{}'.format(episode, l))

    def replay(self):
        self.train_loss.reset_states()
        self.pos_loss.reset_states()
        self.action_loss.reset_states()
        for _ in range(50):
            states, actions, q_values, pos = self.buffer.sample(self.batch_size)
            action_selector = tf.one_hot(actions, self.action_dim)
            with tf.GradientTape() as tape:
                q_pred = self.model(states, training=True)
                action_pred = q_pred[:, :3]
                pos_pred = q_pred[:, 3:]
                action_pred = action_pred * action_selector
                action_pred = tf.reduce_sum(action_pred, axis=1)
                action_loss = tf.losses.mean_squared_error(q_values, action_pred)
                pos_loss = tf.losses.mean_squared_error(pos, pos_pred)
                loss = pos_loss + action_loss
                self.action_loss(action_loss)
                self.pos_loss(pos_loss)
                self.train_loss(loss)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        print('ActionLoss:{}, PosLoss:{}, Train loss:{}'.format(self.action_loss.result().numpy(),
                                                                self.pos_loss.result().numpy(),
                                                                self.train_loss.result().numpy()))

    def find_stop_steps(self, flys):
        steps = []
        for i in range(len(flys)):
            if i > 0 and not flys[i] and flys[i - 1]:  # 不是初始状态，当前是站在台阶，前一步是空中状态。
                steps.append(i + 1)
        steps.append(len(flys))
        return steps

    def compute_jump_value(self, rewards, flys):  # 计算一次跳跃的得分
        q = 0
        q_list = []
        for r, f in zip(rewards[::-1], flys[::-1]):
            if r > 0 and q < 0:
                q_list.append(q)
            else:
                q = r + self.gamma * q
                q_list.append(q)
        return q_list[::-1]

    def computeValueList(self, rewards, flys):
        q_list = []
        steps = self.find_stop_steps(flys)
        i = 0
        for s in steps:
            q_list += self.compute_jump_value(rewards[i:s], flys[i:s])
            i = s
        return q_list

    def train(self):
        for episode in range(self.train_episode):
            total_reward, done = 0, False
            state, x, y = self.env.reset()
            step = 0
            episode_rewards = []
            episode_flys = []
            episode_states = []
            episode_actions = []
            episode_xs = []
            episode_ys = []
            while not done:
                step += 1
                action = self.choose_action(state, episode, x)
                next_state, reward, done, is_fly, next_x, next_y, layers = self.env.step(action)
                episode_rewards.append(reward)
                episode_flys.append(is_fly)
                episode_states.append(state)
                episode_actions.append(action)
                episode_xs.append(x)
                episode_ys.append(y)
                total_reward += reward
                state = next_state
                x = next_x
                y = next_y
            q_list = self.computeValueList(episode_rewards, episode_flys)
            for s, a, r, q, x, y, f in zip(episode_states, episode_actions, episode_rewards, q_list, episode_xs,
                                        episode_ys, episode_flys):
                self.buffer.push(s, a, q, [x, y])
            if len(self.buffer) > self.batch_size:
                self.replay()
                self.train_num += 1
            timestr = time.strftime("%H:%M:%S", time.localtime())
            self.layers_mean(layers)
            print('{}, episode:{},Mean Layers:{}, current layers:{}'.format(timestr, episode,
                                                                            self.layers_mean.result().numpy(), layers))
            if (episode + 1) % 100 == 0:
                print("{}, episode:{}, mean layers:{}".format(timestr, episode, self.layers_mean.result().numpy()))
                self.layers_mean.reset_states()
                self.model.save('model{}'.format((episode + 1) // 50))


# 训练前请先利用CheatEngine工具查找游戏里一些关键数值的内存地址，否则不能正常训练。
if __name__ == '__main__':
    env = NSSHAFT()
    agent = Agent(env)
    agent.train()