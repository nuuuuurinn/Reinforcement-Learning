import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import time

env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 500
max_steps = 100

def choose_action(state):
    print("random floating point:", random.uniform(0, 1))
    if random.uniform(0, 1) < epsilon: #returns random floating point
        print("action space sample:", env.action_space.sample())
        return env.action_space.sample()  #returns 0 or 1
    else:
        print("index of Q state:", np.argmax(Q[state]))
        return np.argmax(Q[state])  # exploit best action based on Q-values, returns index 0 or 1

def train():
    win_count = 0
    for episode in range(episodes):
        state, _ = env.reset()
        state = (state[0], state[1], int(state[2]))
        print("current state:", state)
        done = False
        total_reward = 0    

        for _ in range(max_steps):
            action = choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            print("reward:", reward)
            next_state = (next_state[0], next_state[1], int(next_state[2]))
            print("next state:", next_state)

            best_next_action = np.argmax(Q[next_state])
            print("index of best next action:", best_next_action)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state
            total_reward += reward
            print("total reward:", total_reward)

            if done:
                break

        if total_reward > 0:
            win_count += 1
            print(f"Episode {episode + 1}: Win")
        elif total_reward < 0:
            print(f"Episode {episode + 1}: Loss")
        else:
            print(f"Episode {episode + 1}: Draw")
        print("\n")

        global epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print("epsilon:", epsilon)

    print(f"Final win rate: {win_count / episodes:.4f}")


train()
env.close()
