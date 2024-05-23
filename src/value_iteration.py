import numpy as np
import gymnasium as gym
import pprint
pp = pprint.PrettyPrinter(indent=4)
from tqdm import tqdm
import argparse

from utils import save_heatmap, save_policy, run_policy

class ValueIteration:
    def __init__(self, env:gym.Env, theta:float=0.001, gamma:float=0.99):

        self.theta = theta
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.V = np.random.randn(self.n_states)
        self.V[-1] = 0
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.P = env.P

        self.gamma = gamma

    def evaluate_policy(self):
        delta = self.theta*2
        while delta > self.theta:
            delta = 0
            for state in range(self.n_states):
                    v = np.ones(self.n_actions) * -np.inf

                    for action in range(self.n_actions):
                        transitions_list = env.P[state][action]
                        
                        for i in transitions_list:
                            transition_prob, next_state, reward, done = i
                            v[action] = self.policy[state, action]*transition_prob*(reward + self.gamma*self.V[next_state])

                    v = max(v)
                    delta = max(delta, abs(self.V[state] - v))
                    self.V[state] = v

    
    def determine_policy(self):
        qvalues_matrix=np.zeros((self.n_states,self.n_actions))
        improved_policy=np.zeros((self.n_states,self.n_actions))

        for state in range(self.n_states):
                    for action in range(self.n_actions):
                        transitions_list = env.P[state][action]
                        for i in transitions_list:
                            transition_prob, next_state, reward, done = i

                            qvalues_matrix[state,action]=qvalues_matrix[state,action]+transition_prob*(reward+self.gamma*self.V[next_state])

                    best_action=np.where(qvalues_matrix[state,:]==np.max(qvalues_matrix[state,:]))
                    improved_policy[state,best_action]=1/np.size(best_action)

        self.policy = improved_policy


    def value_iteration(self):
        self.evaluate_policy()
        self.determine_policy()

            


if __name__ == "__main__":
    # Frozen Lake:
    # 0: Move left
    # 1: Move down
    # 2: Move right
    # 3: Move up

    parser = argparse.ArgumentParser()
    parser.add_argument('--show_policy', action='store_true', help='Show policy')
    args = parser.parse_args()

    show_policy = args.show_policy


    rows = 4
    cols = 4
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
    
    policy = ValueIteration(env=env)

    print("Initial Policy:\n", policy.policy, "\n")
    print('Transition Matrix:\n')
    pp.pprint(policy.P)

    policy.value_iteration()

    save_heatmap(policy.V.reshape(rows, cols),"value_iteration_value_function.png")
    save_policy(policy.policy,"value_iteration_policy.png", rows, cols)

    print("Final Value array:\n", policy.V.reshape(rows, cols), "\n")
    print("Final Policy:\n", policy.policy, "\n")

    if show_policy:
        run_policy(env, policy=policy.policy)

    

