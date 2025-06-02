from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def get_scalar_data(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    data = ea.Scalars(tag)
    steps = [x.step for x in data]
    values = [x.value for x in data]
    return steps, values

# Define paths and tag to plot
logdir_24 = './ppo_tensorboard/PPO_24'
logdir_23 = './ppo_tensorboard/PPO_23'
tag = 'rollout/ep_rew_mean'

# Get data
steps_24, values_24 = get_scalar_data(logdir_24, tag)
steps_23, values_23 = get_scalar_data(logdir_23, tag)

# Plot both
plt.plot(steps_24, values_24, label='PPO_24')
plt.plot(steps_23, values_23, label='PPO_23')
plt.xlabel("Steps")
plt.ylabel("Episode Reward Mean")
plt.title("Training Progress Comparison")
plt.legend()
plt.grid(True)
plt.show()
