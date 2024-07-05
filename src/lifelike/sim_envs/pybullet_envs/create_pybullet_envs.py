import gym
from gym.spaces import Tuple as GymTuple
from lifelike.sim_envs.pybullet_envs.primitive_level_env.primitive_level_env import PrimitiveLevelEnv


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        self.observation_space = GymTuple([env.observation_space])
        self.action_space = GymTuple([env.action_space])

    def reset(self, **kwargs):
        obs = self.env.reset()
        return (obs,)

    def step(self, action):
        obs, rwd, done, info = super(SingleAgentWrapper, self).step(action[0])
        return (obs,), (rwd,), done, info


def create_tracking_game(**env_config):
    arena_id = env_config["arena_id"]
    assert arena_id in [
        "LeggedRobotTracking",
    ]

    def create_single_env():
        enable_render = env_config.get("render", False)
        control_freq = env_config.get("control_freq", 25.0)
        sim_freq = env_config.get("sim_freq", 500.0)
        kp = env_config.get("kp", 50.0)
        kd = env_config.get("kd", 1.0)
        max_tau = env_config.get("max_tau", 18.0)

        data_path = env_config.get("data_path", "")
        prop_type = env_config.get("prop_type", "")

        prioritized_sample_factor = env_config.get("prioritized_sample_factor", 0.0)
        set_obstacle = env_config.get("set_obstacle", False)
        obstacle_height = env_config.get("obstacle_height", 0.0)

        reward_weights = env_config.get("reward_weights", None)
        env0 = PrimitiveLevelEnv(
            enable_render=enable_render,
            control_freq=control_freq,
            kp=kp,
            kd=kd,
            foot_lateral_friction=env_config.get('foot_lateral_friction', 0.5),
            max_tau=max_tau,
            sim_freq=sim_freq,
            video_path=env_config.get('video_path', None),
            enable_gui=env_config.get('enable_gui', True),
            data_path=data_path,
            prop_type=prop_type,
            prioritized_sample_factor=prioritized_sample_factor,
            set_obstacle=set_obstacle,
            obstacle_height=obstacle_height,
            reward_weights=reward_weights,
        )

        env0 = SingleAgentWrapper(env0)
        return env0
    env = create_single_env()
    return env


def create_playground_game(**env_config):
    arena_id = env_config["arena_id"]
    assert arena_id in [
        "Playground",
    ]

    def create_single_env():
        if arena_id == "Playground":
            from lifelike.sim_envs.pybullet_envs.max_game_elements.playground_env import PlayGroundEnv
        else:
            raise NotImplementedError("Unknown game id.")
        enable_render = env_config["render"] if "render" in env_config else False
        control_freq = env_config["control_freq"] if "control_freq" in env_config else 50.0
        kp = env_config["kp"] if "kp" in env_config else 50.0
        kd = env_config["kd"] if "kd" in env_config else 1.0
        max_tau = env_config["max_tau"] if "max_tau" in env_config else 16.0
        prop_type = env_config["prop_type"] if "prop_type" in env_config else None
        max_steps = env_config["max_steps"] if "max_steps" in env_config else 1000
        obs_randomization = env_config["obs_randomization"] if "obs_randomization" in env_config else None
        env_randomize_config = env_config["env_randomize_config"] if "env_randomize_config" in env_config else None
        env0 = PlayGroundEnv(
            enable_render=enable_render,
            control_freq=control_freq,
            kp=kp, kd=kd,
            max_tau=max_tau,
            prop_type=prop_type,
            max_steps=max_steps,
            obs_randomization=obs_randomization,
            env_randomize_config=env_randomize_config,
        )
        return env0

    env = create_single_env()
    env = SingleAgentWrapper(env)
    return env


def create_chase_tag_game(**env_config):
    arena_id = env_config["arena_id"]
    assert arena_id in [
        "CTG",
    ]

    def create_single_env():
        if arena_id == "CTG":
            from lifelike.sim_envs.pybullet_envs.max_game.chase_tag_game_env import ChaseTagGameEnv
        else:
            raise NotImplementedError("Unknown game id.")
        enable_render = env_config["render"] if "render" in env_config else False
        control_freq = env_config["control_freq"] if "control_freq" in env_config else 25.0
        kp = env_config["kp"] if "kp" in env_config else 50.0
        kd = env_config["kd"] if "kd" in env_config else 1.0
        max_tau = env_config["max_tau"] if "max_tau" in env_config else 18.0
        prop_type = env_config["prop_type"] if "prop_type" in env_config else None
        max_steps = env_config["max_steps"] if "max_steps" in env_config else 1000
        obs_randomization = env_config["obs_randomization"] if "obs_randomization" in env_config else None
        element_config = env_config.get('element_config', {})
        env_randomize_config = env_config.get('env_randomize_config', {})
        env0 = ChaseTagGameEnv(
            enable_render=enable_render,
            control_freq=control_freq,
            kp=kp, kd=kd,
            max_tau=max_tau,
            prop_type=prop_type,
            max_steps=max_steps,
            obs_randomization=obs_randomization,
            env_randomize_config=env_randomize_config,
            element_config=element_config,
        )

        return env0

    env = create_single_env()
    return env


def create_tracking_env(**env_config):
    env = create_tracking_game(**env_config)
    env.observation_space = env.observation_space.spaces[0]
    env.action_space = env.action_space.spaces[0]
    return env


def create_playground_env(**env_config):
    env = create_playground_game(**env_config)
    env.observation_space = env.observation_space.spaces[0]
    env.action_space = env.action_space.spaces[0]
    return env


def create_chase_tag_env(**env_config):
    env = create_chase_tag_game(**env_config)
    env.observation_space = env.observation_space.spaces[0]
    env.action_space = env.action_space.spaces[0]
    return env
