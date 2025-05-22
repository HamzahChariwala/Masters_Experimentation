import numpy as np

n=5

rotations = {
    0: np.array([[1, 0], [0, 1]]),    # Right → no rotation
    1: np.array([[0, 1], [-1, 0]]),   # Down → rotate right
    2: np.array([[-1, 0], [0, -1]]),  # Left → rotate 180°
    3: np.array([[0, -1], [1, 0]])    # Up → rotate left
}

orientations = [0, 1, 2, 3]

def view_position(n):
    return (n//2, 0)

agent_view_position = view_position(n)

original_agent_position = np.array([5, 10])
sample_array = np.zeros((20,20))

for i in range(n):
    for j in range(n):
        for orientation in orientations:
            # Reset agent position for each cell in the view
            agent_position = original_agent_position.copy()
            
            # Calculate relative position from agent view center
            rel_pos = np.array([j - agent_view_position[1], i - agent_view_position[0]])
            
            # Apply rotation
            rotation = rotations[orientations[1]]
            rotated_pos = rotation @ rel_pos
            
            # Calculate final position
            final_position = agent_position + rotated_pos
            
            # Check bounds before setting value
            x, y = final_position.astype(int)
            if 0 <= x < sample_array.shape[0] and 0 <= y < sample_array.shape[1]:
                sample_array[y][x] = 1

sample_array[original_agent_position[1]][original_agent_position[0]] = 2

print(sample_array)
            
