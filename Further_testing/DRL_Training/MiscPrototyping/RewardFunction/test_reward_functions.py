import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def linear_reward(steps, x_intercept, y_intercept):
    """
    Linear reward function with specified x and y intercepts.
    
    Parameters:
    -----------
    steps : array-like
        Number of steps taken
    x_intercept : float
        Number of steps at which reward becomes 0
    y_intercept : float
        Initial reward value at step 0
    """
    slope = y_intercept / x_intercept
    return np.maximum(0, y_intercept - slope * steps)

def exponential_reward(steps, x_intercept, y_intercept):
    """
    Exponential reward function with specified x and y intercepts.
    
    Parameters:
    -----------
    steps : array-like
        Number of steps taken
    x_intercept : float
        Number of steps at which reward becomes approximately 0
    y_intercept : float
        Initial reward value at step 0
    """
    # Calculate slope to achieve desired x-intercept
    slope = -np.log(0.01) / x_intercept  # 0.01 is our "close enough to zero" threshold
    return y_intercept / np.exp(slope * steps)

def sigmoid_reward(steps, x_intercept, y_intercept, transition_width=10):
    """
    Sigmoid reward function with specified x and y intercepts.
    
    Parameters:
    -----------
    steps : array-like
        Number of steps taken
    x_intercept : float
        Number of steps at which reward becomes approximately 0
    y_intercept : float
        Initial reward value at step 0
    transition_width : float
        Controls how quickly the reward transitions from y_intercept to 0
    """
    # Calculate parameters to achieve desired intercepts
    slope = 4 / transition_width  # Controls the steepness of the transition
    shift = x_intercept / 2  # Center the transition at x_intercept/2
    return y_intercept / (1 + np.exp(slope * (steps - shift)))

def plot_reward_functions():
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate x values (steps)
    steps = np.linspace(0, 150, 1000)
    
    # Test different intercept combinations
    x_intercepts = [50, 100, 150]
    y_intercepts = [1.0, 2.0]
    transition_widths = [5, 10, 20]
    
    # Plot linear rewards
    for x_int in x_intercepts:
        for y_int in y_intercepts:
            rewards = linear_reward(steps, x_int, y_int)
            ax1.plot(steps, rewards, label=f'x={x_int}, y={y_int}')
    ax1.set_title('Linear Reward Function')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot exponential rewards
    for x_int in x_intercepts:
        for y_int in y_intercepts:
            rewards = exponential_reward(steps, x_int, y_int)
            ax2.plot(steps, rewards, label=f'x={x_int}, y={y_int}')
    ax2.set_title('Exponential Reward Function')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    ax2.legend()
    
    # Plot sigmoid rewards
    for x_int in x_intercepts:
        for y_int in y_intercepts:
            for width in transition_widths:
                rewards = sigmoid_reward(steps, x_int, y_int, width)
                ax3.plot(steps, rewards, label=f'x={x_int}, y={y_int}, w={width}')
    ax3.set_title('Sigmoid Reward Function')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('reward_functions.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_reward_functions() 