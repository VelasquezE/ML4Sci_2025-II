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
wanted_point = 0

print(f"Las partículas que pasaron 0.2 unidades de (0, 0) son: \n")

for id in particles:
    for coordinate in trajectories[id]:
        if (coordinate[0] - wanted_point) <= epsilon and (coordinate[1] - wanted_point) <= epsilon:
            print(f"{id}: {coordinate}\n")