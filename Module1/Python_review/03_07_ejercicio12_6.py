import numpy as np

# Given
time_points = [0, 1, 2, 3, 4]  # seconds
particles = {"P1", "P2", "P3"}  # unique IDs
trajectories = {
    "P1": [(1.0, 2.0), (0.5, 0.3), (0.1, -0.1), (0.0, 0.01), (-0.2, -0.3)],
    "P2": [(2.0, 1.0), (1.8, 0.9), (1.5, 0.6), (1.0, 0.2), (0.5, -0.1)],
    "P3": [(-1.0, -1.0), (-0.5, -0.5), (0.0, 0.0), (0.2, 0.1), (0.3, 0.4)]
}

# Task: Find which particles came within 0.2 units of (0,0)

epsilon = 0.2
wanted_point = 0.0
selected_particles = {"P1": False, "P2": False, "P3": False}

print(f"Las partículas que pasaron 0.2 unidades de (0, 0) son:")

for id in particles:
    for coordinate in trajectories[id]:
        if np.abs(coordinate[0] - wanted_point) <= epsilon and np.abs(coordinate[1] - wanted_point) <= epsilon:
            selected_particles[id] = True

for particle in selected_particles:
    if selected_particles[particle] == True:
        print(f"{particle} ")
