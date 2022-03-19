import random
import numpy as np


class ValueReplayBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.p_buffer = ReplayBuffer(self.capacity // 2)  # value 为正的
        self.n_buffer = ReplayBuffer(self.capacity // 2)  # value 为负的

    def __len__(self):
        return min(len(self.p_buffer), len(self.n_buffer))

    def push(self, state, action, value, pos):  # 根据
        if value >= 0:
            self.p_buffer.push(state, action, value, pos)
        else:
            self.n_buffer.push(state, action, value, pos)

    def sample(self, batch_size=32):
        p_state, p_action, p_value, p_pos = self.p_buffer.sample(batch_size // 2)
        n_state, n_action, n_value, n_pos = self.n_buffer.sample(batch_size // 2)
        return map(np.concatenate, ([p_state, n_state], [p_action, n_action], [p_value, n_value], [p_pos, n_pos]))


class ReplayBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, pos):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, pos)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, pos = map(np.stack, zip(*batch))
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, pos
