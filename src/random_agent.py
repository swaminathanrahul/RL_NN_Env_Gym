# Reset
import gymnasium as gym
import numpy as np
import pickle

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()

X = list()
y = list()
number_of_episodes = 1000

for i in range(number_of_episodes):

    done = False  # Reset game state
    state = env.reset()[0]  # For a new run, reset environment

    while not done:
        random_action = env.action_space.sample()  # Choose a random action
        state_p, reward, terminated, truncated, _ = env.step(random_action)  # take a step and get outcome
        X.append([*state.tolist(), int(random_action)])
        y.append(state_p.tolist())
        state = state_p  # s is now s'
        done = terminated or truncated  # check if game over

#print(f"X: {X[:5]}, \n y: {y[:5]}")
data = [X, y]

with open("data/random_agent_data.pkl", "wb") as file:
    file.write(pickle.dumps(data))
    

