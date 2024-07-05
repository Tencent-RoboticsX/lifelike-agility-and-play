import importlib

import joblib
from absl import app
from absl import flags
from lifelike.sim_envs.pybullet_envs.create_pybullet_envs import create_playground_env
from tleague.actors.agent import PGAgent

flags.DEFINE_string("hurdle_model_path", 'data/models/environmental_level_hurdle.model', "the trained EPMC model path")
flags.DEFINE_string("hole_model_path", 'data/models/environmental_level_hole.model', "the trained EPMC model path")
flags.DEFINE_string("cube_model_path", 'data/models/environmental_level_cube.model', "the trained EPMC model path")

FLAGS = flags.FLAGS


def main(_):
    obs_randomization = {}
    env_randomize_config = {
        'element_id': 1,  # 1: hurdle, 2: hole, 3: cube
        'height_range': [0.0, 0.0],
        'friction_range': [0.4, 1.0],
        'disturb_force_config': {
            'start_time': 0.5,
            'interval_time': 1.0,  # default 3.0
            'duration_time': 0.2,  # default 0.2
            'horizontal_force': [0, 50],
            'vertical_force': [0, 10],
        },
        'cmd_vary_freq_range':  [9999, 10000],  # default [25, 200] -> [0.5s, 4s]
        'target_spd_range': [3.0, 3.0],
        'auxiliary_radius': None,
        'hole_config': {
            'min_gap_height': 0.25,
            'max_gap_height': 0.25,
        },
    }
    env_config = {
        'arena_id': 'Playground',
        'render': True,
        'control_freq': 50.0,
        'prop_type': ['joint_pos', 'joint_vel', 'root_ang_vel_loc', 'root_lin_vel_loc', 'e_g'],
        'kp': 50.0,
        'kd': 0.5,
        'max_tau': 16,
        'max_steps': 1000,
        'obs_randomization': obs_randomization,
        'env_randomize_config': env_randomize_config,
    }
    policy_config = {
        'playground_input': True,
        'llc_light': True,
        'batch_size': 1,
        'rollout_len': 1,
        'test': True,
        'use_self_fed_heads': True,
        'use_loss_type': 'none',
        'use_value_head': True,
        'rms_momentum': 0.0001,
        'n_v': 1,
        'main_activate_func': 'relu',
        'append_hist_a': True,
        'use_lstm': True,
        'hs_len': 64 * 3,
        'nlstm': 32,
        'discrete_z': True,
        'z_len': 256,
        'z_len_llc': 32,
        'norm_z': False,
        'expert_lstm': True,
        'isolate_z_logvar': False,
        'bot_neck_z_embed_size': 32,
        'bot_neck_prop_embed_size': 64,
        'lstm_dropout_rate': 0.0,
        'lstm_cell_type': 'lstm',
        'lstm_layer_norm': True,
        'weight_decay': 0.00002,
        'sync_statistics': 'none',
    }
    policy = "lifelike.networks.legged_robot.epmc_net.epmc_net"

    # env, policy and model file
    env = create_playground_env(**env_config)
    ob_space, ac_space = env.observation_space, env.action_space

    policy_module, _ = policy.rsplit(".", 1)
    policy = importlib.import_module(policy_module)
    if env_randomize_config["element_id"] == 1:
        model = joblib.load(FLAGS.hurdle_model_path)
    elif env_randomize_config["element_id"] == 2:
        model = joblib.load(FLAGS.hole_model_path)
    elif env_randomize_config["element_id"] == 3:
        model = joblib.load(FLAGS.cube_model_path)
    else:
        model = None
    # init agent
    agent = PGAgent(
        policy, ob_space, ac_space, policy_config=policy_config, scope_name='agent')
    agent.load_model(model.model)
    obs = env.reset()
    agent.reset(obs[0])

    # evaluate model
    epi_length = 0
    reward_sum = 0
    total_reward = 0
    n_episodes = 0
    while True:
        act = agent.step(obs[0], argmax=True)
        obs, rwd, done, _ = env.step([act])
        reward_sum += rwd[0]
        epi_length += 1
        if done:
            obs = env.reset()
            agent.reset(obs[0])
            n_episodes += 1
            total_reward += reward_sum
            ave_r = total_reward / float(n_episodes)
            print("reward_sum: {}, episode length: {}, n_episodes: {}, average_reward_sum: {:.2f}.".format(
                reward_sum, epi_length, n_episodes, ave_r))
            epi_length = 0
            reward_sum = 0


if __name__ == '__main__':
    app.run(main)
