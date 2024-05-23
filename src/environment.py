import numpy as np
import matplotlib.pyplot as plt
import pprint



class GridWorld:
    def __init__(self, rows, cols, start_state, goal_state, obstacles=[None, None], transition_prob:float=1.0):
        self.rows = rows
        self.cols = cols
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacles = obstacles
        self.state = start_state
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = np.zeros((rows, cols)) - 1
        self.rewards[goal_state] = 1
        # self.rewards[obstacles] = -1
        self.steps = 0
        self.num_states = self.rows * self.cols
        self.num_actions = len(self.actions)
        self.transition_prob = transition_prob
        self.P = {}
        self.create_transition_dynamics()

    def step(self, action):
        action = self.actions[action]
        self.steps += 1
        row, col = self.state

        if action == 'up':
            new_row = row - 1
            new_col = col
        elif action == 'down':
            new_row = row + 1
            new_col = col
        elif action == 'left':
            new_row = row
            new_col = col - 1
        else:  # action == 'right'
            new_row = row
            new_col = col + 1

        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            # Stepped off the edge of the world
            reward = -1.0
            # self.state = self.start_state
            self.reset()
            terminal = True
        else:
            new_state = (new_row, new_col)
            if new_state in self.obstacles:
                reward = self.rewards[new_state]
                # self.state = self.state
                # self.reset()
                terminal = True
            else:
                reward = self.rewards[new_state]
                self.state = new_state
                terminal = self.is_terminal()

        return self.state, reward, terminal

    def reset(self):
        self.steps = 0
        self.state = self.start_state
        return self.state
    
    def is_terminal(self):
        if np.array_equal(self.state, self.goal_state) or self.steps >= 50:
            return True
        else:
            return False
        
    def print_env(self):
        world = self.rewards.copy()
        world[self.goal_state] = 10
        world[self.state] = 5
        print(self.rewards)
        print(world)
        print(self.state, "\n")

        plt.imshow(world, cmap='inferno')
        plt.axis('off')
        plt.show()

    def to_2d_indices(self, unique_index):
        num_cols=self.cols
        i = unique_index // num_cols
        j = unique_index % num_cols
        return (i, j)
    
    def to_unique_index(self, state):
        num_cols=self.cols
        return state[0] * num_cols + state[1]
    
    def create_transition_dynamics(self):
        for state in range(self.num_states):
                self.P[state] = {}
                for action in range(self.num_actions):
                    self.state = self.to_2d_indices(state) 
                    new_state, reward, is_terminal = self.step(action)
                    self.P[state][action] = [(self.transition_prob, self.to_unique_index(new_state) , reward, is_terminal)]

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)
    
    env = GridWorld(4, 4, (3, 3), (0, 0))
    pp.pprint(env.P)


    # for i in range(10):
    #     action = np.random.choice([0,2])
    #     new_state, reward, is_terminal = env.step(action)
    #     print(new_state, reward, is_terminal)
    #     env.print_env()
    