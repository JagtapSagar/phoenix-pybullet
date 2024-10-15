import gymnasium as gym
import time
import phoenix_drone_simulation
import numpy as np
from phoenix_drone_simulation.utils.POMDP import POMDPWrapper

# env = gym.make('DroneHoverBulletEnv-v0', render_mode="human")
# env = gym.make('DroneCircleBulletEnv-v0', render_mode="human")
env = gym.make('DroneLanderBulletEnv-v0', render_mode="human")
POMDP = POMDPWrapper('flicker',0.15)

while True:
    done = False
    x, _ = env.reset()
    while not done:
        # random_action = env.action_space.sample()
        random_action = np.array([0.1,0,0,1.0])
        x, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        time.sleep(0.05)