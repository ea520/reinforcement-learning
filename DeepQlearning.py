import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize
dt = 0.1
possible_actions = np.linspace(-1, 1., 11)
stopping_distance = 0.1
stopping_velocity = 0.05 * stopping_distance / dt
gamma = 1.
terminal_velocity = 3
cd = np.abs(possible_actions).max() / terminal_velocity**2
max_distance = stopping_distance * 100


def update(states, actions):
    x = states[:, 0]
    v = states[:, 1]
    ret = np.zeros_like(states)
    ret[:, 0] = x + v * dt
    ret[:, 1] = v + (actions - cd * v * np.abs(v)) * dt
    # inelastic collisions with imaginary wall at +/- max_distance
    collisions = np.abs(ret[:, 0]) > max_distance
    ret[:, 0][collisions] = np.sign(ret[:, 0][collisions]) * max_distance
    ret[:, 1][collisions] = 0
    return ret


def reward(states, actions):
    r = np.zeros(states.shape[0]) - dt
    new_states = update(states, actions)
    x = new_states[:, 0]
    v = new_states[:, 1]
    stopping_indexes = (np.abs(x) < stopping_distance) & (
        np.abs(v) < stopping_velocity)
    r[stopping_indexes] = 0
    return r


def get_ys(states, actions, Q):
    rewards = reward(states, actions)
    next_states = update(states, actions)
    values = np.zeros(states.shape[0]) - np.inf
    for action in possible_actions:
        values = np.max([values, Q(next_states, action)], axis=0)
    values[rewards == 0] = 0
    return rewards + gamma * values


def get_samples(episode_count, episode_length, policy, x0=None):
    N = episode_count * episode_length
    indexes = np.arange(0, N, episode_length, dtype=int)
    states = np.zeros((N, 2)) - np.inf
    actions = np.zeros(N) - np.inf
    if x0 is None:
        states[indexes, 0] = np.random.uniform(-1, 1,
                                               len(indexes)) * max_distance
        states[indexes, 1] = np.random.uniform(-1, 1,
                                               len(indexes)) * terminal_velocity
    else:
        states[indexes] = x0
    for _ in range(episode_length):
        # out of bounds doesn't matter as we only go to N//2
        actions[indexes] = policy(states[indexes])
        indexes = indexes[indexes + 1 < N]
        states[indexes + 1] = update(states[indexes],
                                     actions[indexes])
        indexes = indexes + 1
    return states, actions


def get_samples_symmetric(episode_count, episode_length, policy):
    states, actions = get_samples(episode_count // 2, episode_length, policy)
    states = np.concatenate([states, -states])
    actions = np.concatenate([actions, -actions])
    return states, actions


def get_Q(model):
    def Q(states, actions):
        inputs = np.empty((len(states), 3))
        inputs[:, :-1] = states
        inputs[:, -1] = actions

        ret = model(inputs)
        if len(ret.shape) > 1:
            ret = ret[:, 0].numpy()
            return ret
    return Q


def get_policy(Q, epsilon):
    def policy(states):
        random_indexes = np.random.rand(len(states)) < epsilon
        random_actions = np.random.choice(
            possible_actions, np.count_nonzero(random_indexes))

        non_random_actions = np.zeros(
            len(states) - len(random_actions)) - np.inf
        non_random_values = np.zeros(len(non_random_actions)) - np.inf
        non_random_states = states[np.logical_not(random_indexes)]
        for a in possible_actions:
            new_values = Q(non_random_states, a)
            indexes = np.where(new_values > non_random_values)[0]
            equal = np.where(new_values == non_random_values)[0]
            indexes = np.concatenate(
                (indexes, equal[np.random.rand(len(equal)) < 0.5]))
            non_random_actions[indexes] = a
            non_random_values[indexes] = new_values[indexes]

        ret = np.concatenate((non_random_actions, random_actions))
        return ret
    return policy


def get_policy_continuous(Q):
    def policy(state):
        def objective(action):
            return -Q(state, action)
        ret = scipy.optimize.minimize_scalar(
            objective, 0., bounds=(-1, 1), method='bounded')
        return ret.x

    return policy


def train(_model=None, epsilon=1):
    if _model is None:
        _model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(3,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        loss_fn = tf.keras.losses.MeanSquaredError()
        _model.compile(
            optimizer='adam',
            loss=loss_fn
        )
    t0 = time.time()
    for i in range(1, 10):
        episode_count = 200
        episode_length = 100
        Q = get_Q(_model)
        policy = get_policy(Q, epsilon)
        states, actions = get_samples(
            episode_count, episode_length, policy)

        ys = get_ys(states, actions, Q)
        x_train = np.zeros((len(actions), 3)) - np.inf
        x_train[:, :-1] = states
        x_train[:, -1] = actions
        _model.fit(x_train, ys, epochs=5, verbose=0)
        final_epsilon = 0.0
        epsilon = 0.9 * (epsilon - final_epsilon) + final_epsilon
        print(
            f"{i:3}{epsilon:6.2f}{-np.mean(reward(states, actions)):6.2f} s{(time.time() - t0) / 60.:6.1f} min", flush=True)
    return _model, Q


if __name__ == "__main__":
    model, Q = train()
    model.save("_model.h5")
