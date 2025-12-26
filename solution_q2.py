import gymnasium as gym
import numpy as np
from collections import defaultdict

def collect_random_data(env, num_episodes=1000, max_steps=200):
    counts_sas = defaultdict(int)
    counts_sa = defaultdict(int)
    rewards_sas = defaultdict(float)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    for _ in range(num_episodes):
        s, _ = env.reset()
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            a = env.action_space.sample()
            s_next, r, terminated, truncated, _ = env.step(a)

            counts_sa[(s, a)] += 1
            counts_sas[(s, a, s_next)] += 1
            rewards_sas[(s, a, s_next)] += r

            s = s_next
            steps += 1

    return counts_sas, counts_sa, rewards_sas, n_states, n_actions

def build_model(counts_sas, counts_sa, rewards_sas, n_states, n_actions, smoothing =1e-6):
    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        for a in range(n_actions):
            sa_count = counts_sa.get((s, a), 0)

            if sa_count == 0:
                T[s, a, :] = 1.0 / n_states
                continue
            for (s_k, a_k, s_next), cnt in counts_sas.items():
                if s_k == s and a_k == a:
                    T[s, a, s_next] = cnt / sa_count
                    R[s, a, s_next] = rewards_sas[(s, a, s_next)] / cnt
            T[s, a, :] = (T[s, a, :] + smoothing)
            T[s, a, :] /= T[s, a, :].sum()

    return T, R

def value_iteration(T, R, gamma=0.99, tol = 1e-6, max_iter = 10000):
    n_states, n_actions, _ = T.shape
    V = np.zeros(n_states)

    for _ in range(max_iter):
        V_old = V.copy()
        Q = np.zeros((n_states, n_actions))

        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = np.sum(T[s, a, :] * (R[s, a, :] + gamma * V_old))

        V = np.max(Q, axis = 1)

        if np.max(np.abs(V - V_old)) < tol:
            break

    return V

def extract_policy(T, R, V, gamma=0.99):
    n_states, n_actions, _ = T.shape
    policy = np.zeros(n_states, dtype = int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = np.sum(T[s, a, :] * (R[s, a, :] + gamma * V))
        policy[s] = np.argmax(action_values)

    return policy
    
def run_policy(env, policy, num_episodes=10, max_steps=200):
    for ep in range(num_episodes):
        s, _ = env.reset()
        terminated = truncated = False
        steps = 0
        ep_return = 0

        while not (terminated or truncated) and steps < max_steps:
            a = policy[s]
            s, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            steps += 1

        print(f"Episode {ep + 1} return = {ep_return}")


def main():

    #no GUI
    env_collect = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    env_demo = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")

    counts_sas, counts_sa, rewards_sas, n_states, n_actions = collect_random_data(env_collect, num_episodes=1000)

    T, R = build_model(counts_sas, counts_sa, rewards_sas, n_states, n_actions)

    V = value_iteration(T, R, gamma = 0.99)
    print("Value function:", np.round(V, 4))

    policy = extract_policy(T, R, V)
    print("Optimal policy:", policy)

    print("Running policy in GUI environment:")
    run_policy(env_demo, policy)

    env_collect.close()
    env_demo.close()


if __name__ == "__main__":
    main()
