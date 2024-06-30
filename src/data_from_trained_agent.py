"""Your first learning agent."""

import gymnasium as gym
from ray.rllib.algorithms.dqn import DQNConfig
import numpy as np
import os
import pickle

config = DQNConfig()
# 1.2 - Examine the config by converting it to a dict via .to_dict() method
config_as_dict = config.to_dict()

# 1.3 - Modify the config if needed, e.g. change the "num_gpus" to 0, or change the learning_rate
# (lr)
# 1.4 - Introduce the environment to the agent's config
config.environment(env="CartPole-v1").framework(framework="tf")
# 1.5 - Build the agent from the config with .build()

agent = config.build()
# 2 - Train the agent for one training round with .train and get the reports
reports = agent.train()
print(reports)

# 3 - Run a loop for nr_trainings = 50 times
nr_trainings = 20  # pylint: disable=invalid-name
for _ in range(nr_trainings):
    reports = agent.train()
    print(_, reports["episode_reward_mean"])

# 4 - Visualize the trained agent; This is similar to running the random_agent,
# except that this time we have a trained agent
# 4.1 - Create an environment similar to the training env.
env = gym.make("CartPole-v1", render_mode="rgb_array")

X = list()
y = list()
number_of_episodes = 0

for i in range(number_of_episodes):

    state, _ = env.reset()
    done = False  # pylint: disable=invalid-name

    while not done:
        action = agent.compute_single_action(observation=state, explore=False)
        state_p, reward, terminated, truncated, _ = env.step(action=action)
        X.append([*state.tolist(), int(action)])
        y.append(state_p.tolist())
        state = state_p  # s is now s'
        done = terminated or truncated

X = np.array(X)
y = np.array(y)
data = [X, y]

path = os.path.join("../data", "trained_agent_data_np.pkl")
with open(path, "wb") as file:
    file.write(pickle.dumps(data))
