import numpy as np
import gym
import matplotlib.pyplot as plt
import glob
import imageio
# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0', render_mode="rgb_array")
env.reset()

weights_file = max(glob.glob("weights*.npy")) # most recent file 

Q = np.load(weights_file)

def run(env):
    done = False
    truncated = 0
    state, _ = env.reset()  # state as float

    # Discretize state (the index in the array)
    state_adj = (state - env.observation_space.low) * np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)
    image_list = []
    total_reward = 0
    while not done and not truncated:
        img = env.render()
        image_list.append(img)
        action = np.argmax(Q[state_adj[0], state_adj[1]])
        state, reward, done, truncated, _ = env.step(action)
        state_adj = (state - env.observation_space.low) * np.array([10,100])
        state_adj = (state_adj + 0.5).astype(int)
        total_reward += reward
    print(total_reward)
    return image_list
image_list = run(env)

imageio.mimsave("mountain_car.gif", image_list, fps=10)