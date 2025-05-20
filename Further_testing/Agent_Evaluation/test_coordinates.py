import gymnasium as gym
from Environment_Tooling.BespokeEdits.CustomWrappers import SelectiveObservationWrapper

env = gym.make('MiniGrid-Empty-16x16-v0')
env = SelectiveObservationWrapper(env, cnn_keys=[], mlp_keys=['lava_mask', 'new_image'])

env.reset()
