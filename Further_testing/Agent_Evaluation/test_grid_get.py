import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

def test_grid_get(env_id="MiniGrid-LavaCrossingS11N5-v0", seed=81102):
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    grid = base.grid
    width, height = grid.width, grid.height
    print(f"Grid size: {width}x{height}")
    print("Testing grid.get(x, y) vs grid.get(y, x):")
    
    # Test a few positions: corners, center, and asymmetric
    test_positions = [
        (0, 0),
        (width-1, 0),
        (0, height-1),
        (width-1, height-1),
        (width//2, height//2),
        (2, 5),
        (5, 2),
        (3, 7),
        (7, 3)
    ]
    for x, y in test_positions:
        cell_xy = grid.get(x, y)
        cell_yx = grid.get(y, x)
        print(f"Position (x={x}, y={y}):")
        print(f"  grid.get(x, y): {cell_xy.type if cell_xy else None}")
        print(f"  grid.get(y, x): {cell_yx.type if cell_yx else None}")
    
    # Also print agent's starting position
    agent_x, agent_y = base.agent_pos
    print(f"\nAgent starting position: (x={agent_x}, y={agent_y})")
    cell_agent_xy = grid.get(agent_x, agent_y)
    cell_agent_yx = grid.get(agent_y, agent_x)
    print(f"  grid.get(agent_x, agent_y): {cell_agent_xy.type if cell_agent_xy else None}")
    print(f"  grid.get(agent_y, agent_x): {cell_agent_yx.type if cell_agent_yx else None}")
    
    env.close()

if __name__ == "__main__":
    test_grid_get() 