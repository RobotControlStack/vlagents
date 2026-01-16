import gymnasium as gym


class HumanCameraWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env) -> None:
        self.base_env = env.unwrapped
        super().__init__(env)
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        # ret = dict()
        if not hasattr(self.env, "_has_reset") or not self.env._has_reset:
            self.env.reset()
            self.env._has_reset = True
        # observation["sensor_data"]["human_camera"] = dict(rgb=self.env.render())
        observation["sensor_data"]["base_camera"] = dict(rgb=self.env.render())
        return observation


WRAPPERS = {"HumanCameraWrapper": HumanCameraWrapper}
