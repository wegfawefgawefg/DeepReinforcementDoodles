import numpy as np

class ReplayBuffer:
    def __init__(size, input_shape):
        self.size = size
        self.input_shape = input_shape

        self.states  = np.array((size, *input_shape), dtype=np.float)
        self.actions = np.array( size               , dtype=np.long )
        self.rewards = np.array( size               , dtype=np.float)
        self.states_ = np.array((size, *input_shape), dtype=np.float)
        self.dones   = np.array( size,                dtype=np.bool )

        self.count = 0

    def save(self, state, action, reward, state_, done):
        index = self.count % self.size

        self.states[index]  = state 
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = state_
        self.dones[index]   = done

        self.count += 1

    def sample(self, batch_size):
        end_index = max(self.count, self.size)
        indices = np.random.randint(end_index, batch_size)

        s_states  = self.states[indices]
        s_actions = self.actions[indices]
        s_rewards = self.rewards[indices]
        s_states_ = self.states_[indices]
        s_dones   = self.dones[indices]
        return s_states, s_actions, s_rewards, s_states_, s_dones

