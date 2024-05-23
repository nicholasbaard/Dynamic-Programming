import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
    
def save_heatmap(array:np.array, name:str):

    plt.imshow(array, cmap='coolwarm', interpolation='nearest')
    plt.title("Optimal Value Function")
    plt.savefig(f"../plots/{name}")

def save_policy(array:np.array, name:str, rows:int, cols:int):

    optimal_actions = np.zeros(array.shape[0])

    for i in range(len(array)):
        optimal_actions[i] = np.argmax(array[i])

    optimal_actions = optimal_actions.reshape((rows,cols))

    plot_annotated_grid(optimal_actions, name)

def plot_annotated_grid(arr, name):
    fig, ax = plt.subplots()
    ax.imshow(arr, cmap='coolwarm')
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.annotate(str(arr[i, j]), (j, i), color='white', ha='center', va='center')

    plt.title("Optimal Policy")
    plt.savefig(f"../plots/{name}")

def run_policy(env:Env, policy:np.array):
    
        observation, info = env.reset()
        for _ in range(50):
            env.render()
            action = np.argmax(policy[observation])  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()



