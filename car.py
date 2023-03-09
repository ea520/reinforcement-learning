import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DeepQlearning import get_policy_continuous, get_Q, get_samples, stopping_distance, stopping_velocity, dt, max_distance, terminal_velocity
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

model = tf.keras.models.load_model("model2d6.h5")

Q = get_Q(model)
policy = get_policy_continuous(Q)

states, actions = get_samples(
    1, 200, policy, [0, -terminal_velocity])
time = np.arange(len(states)) * dt
fig, ax = plt.subplots(3)
ax[0].plot(time, states[:, 0])
ax[0].plot(time, states[:, 0] * 0 + stopping_distance, "r--")
ax[0].plot(time, states[:, 0] * 0 - stopping_distance, "r--")
ax[1].plot(time, states[:, 1])
ax[1].plot(time, states[:, 1] * 0 + stopping_velocity, "r--")
ax[1].plot(time, states[:, 1] * 0 - stopping_velocity, "r--")
ax[2].plot(time, actions)

ax[0].set_title("position")
ax[1].set_title("velocity")
ax[2].set_title("acceleration")
fig.tight_layout()


fig2, ax2 = plt.subplots()

width = 1
height = 0.5
ax2.set_aspect("equal", "box")
ax2.set_xlim((-max_distance, max_distance))
ax2.set_ylim((-1, 1))
car = Rectangle((0, 0), 0, 0)
finish = Rectangle((-stopping_distance - width / 2, -1), 2 *
                   stopping_distance + width, 2)
finish.set_zorder(0)
car.set_zorder(1)


def init():
    ax2.add_patch(car)
    finish.set_facecolor("gray")
    ax2.add_patch(finish)
    return car,


def update(frame):
    car.set_width(width)
    car.set_height(height)
    car.set_xy((states[frame, 0] - width / 2, 0 - height / 2))
    return car,


ani = FuncAnimation(
    fig2, update, init_func=init, frames=len(states), interval=10, blit=True)

ani.save("car3.gif", fps=30)
plt.show()
