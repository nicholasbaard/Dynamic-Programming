import numpy as np
import gymnasium as gym
import pprint
pp = pprint.PrettyPrinter(indent=4)
from tqdm import tqdm
import argparse

from utils import save_heatmap, save_policy, run_policy

class PolicyIteration:
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
                    v = 0

                    for action in range(self.n_actions):
                        transitions_list = env.P[state][action]
                        
                        for i in transitions_list:
                            transition_prob, next_state, reward, done = i
                            v += self.policy[state, action]*transition_prob*(reward + self.gamma*self.V[next_state])

                    delta = max(delta, abs(self.V[state] - v))
                    self.V[state] = v

    
    def improve_policy(self):
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


    def policy_iteration(self, max_iterations:int=10):

        for i in tqdm(range(max_iterations), desc="Policy Iteration"):

            if i == 0:
                current_policy=self.policy.copy()

            self.evaluate_policy()
            self.improve_policy()

            # TODO: Check if policy has converged
            if np.array_equal(current_policy, self.policy):
                current_policy=self.policy.copy()
                break
            current_policy=self.policy.copy()

            


if __name__ == "__main__":
    # Frozen Lake:
    # 0: Move left
    # 1: Move down
    # 2: Move right
    # 3: Move up

    parser = argparse.ArgumentParser()
    parser.add_argument('--show_policy', action='store_true', help='Show policy')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum number of iterations for policy iteration')
    args = parser.parse_args()

    show_policy = args.show_policy
    max_iterations = args.max_iterations

    rows = 4
    cols = 4
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
    
    policy = PolicyIteration(env=env)

    print("Initial Policy:\n", policy.policy, "\n")
    print('Transition Matrix:\n')
    pp.pprint(policy.P)

    policy.policy_iteration(max_iterations=max_iterations)

    save_heatmap(policy.V.reshape(rows, cols),"policy_iteration_value_function.png")
    save_policy(policy.policy,"policy_iteration_policy.png", rows, cols)
    
    print("Final Value array:\n", policy.V.reshape(rows, cols), "\n")
    print("Final Policy:\n", policy.policy, "\n")

    if show_policy:
        run_policy(env, policy=policy.policy)

    

