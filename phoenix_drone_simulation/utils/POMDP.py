import torch
import numpy as np


class POMDPWrapper():
    def __init__(self, pomdp='flicker', pomdp_prob = 0.1):
        
        self.pomdp = pomdp
        self.flicker_prob = pomdp_prob
        self.random_noise_sigma = pomdp_prob
        self.range = (1-self.random_noise_sigma, 1 + self.random_noise_sigma)

        if self.pomdp == 'flicker':
            self.prob = self.flicker_prob
        elif self.pomdp == "random_noise":
            self.prob = self.random_noise_sigma
        elif self.pomdp == "flickering_and_random_noise":
            self.flicker_prob = 0.1
            self.prob = pomdp_prob
        else:
            raise ValueError("pomdp was not in ['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing']!")


    def observation(self, obs):
        if self.pomdp == 'flicker':
            if torch.rand(1) <= self.flicker_prob:
                return np.zeros(obs.shape)
            else:
                return obs
        elif self.pomdp == "random_noise":
            noise = torch.FloatTensor(*obs.shape).uniform_(*self.range).numpy()
            return (obs * noise)
        elif self.pomdp == 'flickering_and_random_noise':
            # Flickering
            if torch.rand(1) <= self.flicker_prob:
                new_obs = torch.zeros(obs.shape)
            else:
                new_obs = obs
            noise = torch.FloatTensor(*obs.shape).uniform_(*self.range)
            # Add random noise
            return (new_obs * noise).numpy()
        else:
            raise ValueError("POMDP was not in ['flicker_random', 'flicker_duration', 'flicker_freq', 'random_noise']!")

