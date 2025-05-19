import numpy as np
import gymnasium as gym

class ForceStartState(gym.Wrapper):
    """
    Wrapper that lets you force the agent to spawn at a given position and orientation
    on the very next reset(). Call `.force(pos, dir)` before `.reset()` to queue it up.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._forced_state = None

    def force(self, agent_pos: tuple[int, int], agent_dir: int):
        """
        Queue up a forced start state.
        
        Args:
            agent_pos: (x, y) grid coordinates
            agent_dir: orientation integer (0=right,1=down,2=left,3=up)
        """
        # store as tuple/list to ensure JSON‐serializable if you ever log it
        self._forced_state = (tuple(agent_pos), int(agent_dir))

    def reset(self, *, seed=None, options=None):
        # 1) do the normal reset
        obs, info = super().reset(seed=seed, options=options)

        # 2) if we have a forced state pending, apply it
        if self._forced_state is not None:
            pos, direction = self._forced_state
            base = self.unwrapped  # drill down to the raw MiniGrid env

            # set the env’s internal state
            base.agent_pos = np.array(pos, dtype=int)
            base.agent_dir = direction

            # recompute the very first observation
            if hasattr(base, "gen_obs"):
                obs = base.gen_obs()
            elif hasattr(base, "_get_obs"):
                obs = base._get_obs()
            else:
                # fallback: sample from obs space (unlikely needed)
                obs = self.observation_space.sample()

            # clear for next episode
            self._forced_state = None

        # # 1) pull out the options:
        # if options and 'agent_pos' in options:
        #     self.minigrid_env.agent_pos = np.array(options['agent_pos'])
        #     self.minigrid_env.agent_dir = options['agent_dir']

        # # 2) now *immediately* recompute your features for this very first state:
        # self._update_obs_with_position(obs, info)

        return obs, info
