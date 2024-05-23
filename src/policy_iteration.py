import numpy as np
from environment import GridWorld
import gymnasium as gym
import pprint
from tqdm import tqdm

class PolicyIteration:
    def __init__(self, env:gym.Env, theta:float=0.0001, gamma:float=0.99):

        self.theta = theta
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.V = np.zeros(self.n_states)
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
        qvaluesMatrix=np.zeros((self.n_states,self.n_actions))
        improvedPolicy=np.zeros((self.n_states,self.n_actions))

        for state in range(self.n_states):
                    for action in range(self.n_actions):
                        transitions_list = env.P[state][action]
                        for i in transitions_list:
                            transition_prob, next_state, reward, done = i

                            qvaluesMatrix[state,action]=qvaluesMatrix[state,action]+transition_prob*(reward+self.gamma*self.V[next_state])

                    bestActionIndex=np.where(qvaluesMatrix[state,:]==np.max(qvaluesMatrix[state,:]))
                    improvedPolicy[state,bestActionIndex]=1/np.size(bestActionIndex)

        self.policy = improvedPolicy


    def policy_iteration(self, max_iterations:int=1000):

        for i in tqdm(range(max_iterations), desc="Policy Iteration"):

            if (i == 0):
                currentPolicy=self.policy.copy()

            self.evaluate_policy()
            # print("current policy:")
            # print(currentPolicy)
            self.improve_policy()
            # print("improved policy:")
            # print(self.policy)

            # TODO: Check if policy has converged
            if np.allclose(currentPolicy, self.policy):
                currentPolicy=self.policy
                # print("Policy iteration algorithm converged!")
                break
            currentPolicy=self.policy

            


if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)
    # rows = 4
    # cols = 4
    # env = GridWorld(rows=4, cols=4, start_state=(3, 3), goal_state=(0, 0))
    
    # desc = ["SFFH", 
    #         "FHFH", 
    #         "FHFH", 
    #         "HFFG"]
    # Frozen Lake:
    # 0: Move left
    # 1: Move down
    # 2: Move right
    # 3: Move up

    rows = 4
    cols = 4
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

    # Cliffwalking:
    # 0: Move up
    # 1: Move right
    # 2: Move down
    # 3: Move left

    # rows = 4
    # cols = 12
    # env = gym.make('CliffWalking-v0', render_mode="human")
    
    policy = PolicyIteration(env=env)

    print("value array:\n", policy.V, "\n")

    print("Initial Policy:\n", policy.policy.reshape(policy.n_states, policy.n_actions) , "\n")

    print('Transition Matrix:\n')
    pp.pprint(policy.P)

    policy.policy_iteration(max_iterations=10)
    # policy.evaluate_policy()

    print("Final Value array:\n", policy.V.reshape(rows, cols), "\n")

    # policy.improve_policy()
    # policy.improve_policy()

    print("Final Policy:\n", policy.policy.reshape(policy.n_states, policy.n_actions), "\n")

    observation, info = env.reset()

    for _ in range(50):
        env.render()
        action = np.argmax(policy.policy[observation])  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    

