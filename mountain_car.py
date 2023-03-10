import numpy as np
import gym
import matplotlib.pyplot as plt
import datetime
# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function


def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high -
                  env.observation_space.low) * np.array([10, 100])
    num_states = np.round(num_states, 0).astype(
        int) + 1  # In this state space there are 19 positions and 15 velocities

    # Initialize Q table
    Q = np.random.uniform(
        low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))
    # Table of 19 * 15 * 3 state-action pairs

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon
    reduction = 0.999

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done, truncated = False, False
        tot_reward, reward = 0, 0
        state, _ = env.reset()  # state as float

        # Discretize state (the index in the array)
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not done and not truncated:
            # Render environment for last five episodes
            if i >= (episodes - 20):
                env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, truncated, info = env.step(action)

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * \
                np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward

            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0],
                                                        state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        epsilon = reduction * epsilon + min_eps * (1 - reduction)

        # Track rewards
        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (i + 1) % 100 == 0:
            print(
                f'Episode {i + 1}, Average Reward: {ave_reward}, epsilon: {epsilon}')

    env.close()
    return ave_reward_list, Q


# Run Q-learning algorithm
rewards, Q = QLearning(env, 0.2, 1., 1., 0.0, 10_000)
np.save(f"weights-{datetime.datetime.now().isoformat(timespec='seconds')}", Q)
# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.close()
