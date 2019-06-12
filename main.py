import gym
from tqdm import tqdm
import numpy as np
import random

env = gym.make('CartPole-v0')
max_episodes = 10000
decay = 0.999
epsilon = 1
q = np.zeros((49, 200, 41, 200, 2))
alpha = 0.1
gamma = 1

pbar = tqdm(range(max_episodes), ascii=" .oO0", bar_format="{l_bar}{bar}|{postfix}")


def format(st):
    st[0] = round(st[0], 1)
    st[0] *= 10
    st[0] += 24

    st[1] = round(st[1], 1)
    st[1] *= 10
    st[1] += 100

    st[2] = round(st[2], 1)
    st[2] *= 10
    st[2] += 20

    st[3] = round(st[3], 1)
    st[3] *= 10
    st[3] += 100

    integer = st.astype(int)

    return integer


for ep in range(max_episodes):
    pbar.update(1)
    pbar.set_postfix(Epsilon=str(round(epsilon, 2)))
    state = format(env.reset())
    epsilon *= decay

    if epsilon < 0.1:
        epsilon = 0.1

    while True:

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state[0], state[1], state[2], state[3]])

        if ep % 1000 == 0:
            env.render()
        else:
            env.close()

        new_state, reward, _, _ = env.step(action)

        new_state = format(new_state)

        if new_state[2] < 0 or new_state[2] > 40:
            break

        if new_state[0] < 0 or new_state[0] > 48:
            break

        q[state[0], state[1], state[2], state[3], action] += alpha * (reward + (gamma * (np.max(q[new_state[0], new_state[1], new_state[2], new_state[3]]) - q[state[0], state[1], state[2], state[3], action])))

        state = new_state

