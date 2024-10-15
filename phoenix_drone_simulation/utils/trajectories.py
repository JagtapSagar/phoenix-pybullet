import math
import torch
import random
import numpy as np

def random_trajectory_generator(num_points:int = 100):
    choices = [lemniscate, circle] #, square]
    choice = random.choice(choices)
    return  np.hstack((choice(num_points=num_points), np.zeros((num_points,1))))

def lemniscate(a: float = math.sqrt(2), num_points: int = 200, limits: float = 0.5):
    # Parameters for the lemniscate curve
    # a = Max x-axis value 
    # num_points = Number of points to generate

    # Vary lemniscate size
    a += np.random.uniform(-limits,limits)

    # Generate theta values
    theta = torch.linspace(-math.pi / 2, 3 * math.pi / 2, num_points)

    # Calculate x and y coordinates for the lemniscate curve
    x = a * torch.cos(theta) / (torch.sin(theta) ** 2 + 1)
    y = a * torch.cos(theta) * torch.sin(theta) / (torch.sin(theta) ** 2 + 1)

    return torch.vstack((x, y)).T.numpy()

def circle(r: float = math.sqrt(2), num_points: int = 200, limits: float = 0.5):
    angle_step = 360 / num_points  # Angle increment between waypoints
    waypoints = []

    # Vary radius of the circle
    r += np.random.uniform(-limits,limits)

    for i in range(num_points):
        angle = math.radians(i * angle_step)  # Convert angle to radians
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        waypoints.append((x, y))

    return torch.tensor(waypoints).numpy()

def square(side_length:float = 5,num_points:int = 8, limits: float = 0.5):
    if num_points < 4:
        raise ValueError("A square needs at least 4 waypoints.")
    
    # Calculate the number of waypoints per side
    waypoints_per_side = num_points // 4

    # Vary side length of square
    side_length += np.random.uniform(-limits,limits)
    
    # Calculate the increment in x and y for each waypoint
    x_increment = side_length / (waypoints_per_side - 1)
    y_increment = side_length / (waypoints_per_side - 1)
    
    waypoints = []
    
    # Generate waypoints for the top side
    for i in range(waypoints_per_side):
        waypoints.append((i * x_increment, 0))
    
    # Generate waypoints for the right side
    for i in range(1, waypoints_per_side):
        waypoints.append((side_length, i * y_increment))
    
    # Generate waypoints for the bottom side
    for i in range(1, waypoints_per_side):
        waypoints.append((side_length - i * x_increment, side_length))

    # Generate waypoints for the left side
    for i in range(1, waypoints_per_side - 1):
        waypoints.append((0, side_length - i * y_increment))
    
    return (-(torch.tensor(waypoints) - (side_length/2))).numpy()


