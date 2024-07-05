import importlib

import joblib
from absl import app
from absl import flags
from lifelike.sim_envs.pybullet_envs.create_pybullet_envs import create_chase_tag_env
from tleague.actors.agent import PGAgent

flags.DEFINE_string("model_path", 'data/models/strategic_level.model', "the trained SEPMC model path")
FLAGS = flags.FLAGS


def main(_):
    obs_randomization = {}
    env_randomize_config = {
        'friction_range': [0.4, 1.0],  # range follows ETH's setting [0.4, 1.0]
        'disturb_force_config': {
            'start_time': 0.5,
            'interval_time': 1.0,  # default 3.0
            'duration_time': 0.2,  # default 0.2
            'horizontal_force': [0, 50],  # ETH 50N
            'vertical_force': [0, 10],
        },
        'control_spd': 1.0,
    }
    env_config = {
        'arena_id': 'CTG',
        'render': True,
        'control_freq': 50.0,
        'prop_type': ['joint_pos', 'joint_vel', 'root_ang_vel_loc', 'root_lin_vel_loc', 'e_g'],
        'kp': 50.0,
        'kd': 0.5,
        'max_tau': 16,
        'max_steps': 1000,
        'obs_randomization': obs_randomization,
        'env_randomize_config': env_randomize_config,
        'element_config': {
            'rand_cube': False,
            'hurdle': False,
            'hole': False,
        },
    }
    policy_config = {
        'outer_control_spd': True,
        'take_percep_1d': True,
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
        'hs_len': 64 * 4,
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
    policy = "lifelike.networks.legged_robot.sepmc_net.sepmc_net"

    # env, policy and model file
    env = create_chase_tag_env(**env_config)
    ob_space, ac_space = env.observation_space, env.action_space

    policy_module, _ = policy.rsplit(".", 1)
    policy = importlib.import_module(policy_module)

    if FLAGS.model_path is not None:
        model = joblib.load(FLAGS.model_path)

    # init agent
    agents = [PGAgent(policy, ob_space, ac_space, policy_config=policy_config, scope_name='agent'+str(_))
              for _ in range(2)]
    if FLAGS.model_path is not None:
        [agent.load_model(model.model) for agent in agents]
    obs = env.reset()
    [agent.reset(ob) for ob, agent in zip(obs, agents)]

    # evaluate model
    while True:
        act = [agent.step(ob, argmax=True) for ob, agent in zip(obs, agents)]
        obs, _, done, info = env.step(act)
        if done:
            # break
            print(info)
            obs = env.reset()
            [agent.reset(ob) for ob, agent in zip(obs, agents)]


if __name__ == '__main__':
    app.run(main)
