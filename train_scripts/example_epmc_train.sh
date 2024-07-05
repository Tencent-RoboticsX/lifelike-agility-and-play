# Usage:
#   1. open four terminals;
#   2. "bash example_epmc_train.sh model_pool" in terminal 1;
#   3. "bash example_epmc_train.sh league_mgr" in terminal 2;
#   4. "bash example_epmc_train.sh actor" in terminal 3;
#   5. "bash example_epmc_train.sh learner" in terminal 4;


role=$1
# common args
actor_type=PPO
outer_env=lifelike.sim_envs.pybullet_envs.create_playground_game
outer_env_2=lifelike.sim_envs.pybullet_envs.create_playground_env

game_mgr_type=tleague.game_mgr.game_mgrs.SelfPlayGameMgr && \
game_mgr_config="{
  'max_n_players': 1}"
mutable_hyperparam_type=ConstantHyperparam
hyperparam_config_name="{ \
  'learning_rate': 0.00001, \
  'lam': 0.95, \
  'gamma': 0.95, \
}" && \
policy=lifelike.networks.legged_robot.epmc_net.epmc_net
learner_policy_config="{ \
  'playground_input': True,
  'llc_light': True,
  'test': False,
  'use_self_fed_heads': False,
  'use_loss_type': 'ppo',
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
}" && \
actor_policy_config="{ \
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
}" && \
learner_config="{ \
  'vf_coef': 1, \
  'max_grad_norm': 0.5, \
  'distill_coef': 0.0, \
  'ent_coef': 0.00000, \
  'ep_loss_coef': {}, \
}" && \
env_config="{ \
  'arena_id': 'Playground', \
  'render': False, \
  'control_freq': 50.0, \
  'prop_type': ['joint_pos', 'joint_vel', 'root_ang_vel_loc', 'root_lin_vel_loc', 'e_g'], \
  'kp': 50.0, \
  'kd': 0.5, \
  'max_tau': 16, \
  'max_steps': 1000, \
  'obs_randomization': {}, \
  'env_randomize_config': { \
      'element_id': 0, \
      'height_range': [0.0, 0.0], \
      'friction_range': [0.4, 3.0], \
      'disturb_force_config': { \
        'start_time': 0.5, \
        'interval_time': 1.0, \
        'duration_time': 0.2, \
        'horizontal_force': [0, 50], \
        'vertical_force': [0, 10],}, \
      'cmd_vary_freq_range': [9999, 10000], \
      'target_spd_range': [0.5, 3.0], \
      'auxiliary_radius': 0.02, \
      'hole_config': { \
            'min_gap_height': 0.25, \
            'max_gap_height': 0.25, \
        }, \
      }, \
}" && \
interface_config="{}"

echo "Running as ${role}"

if [ $role == model_pool ]
then
# model pool
python -i -m tleague.bin.run_model_pool \
  --ports 10003:10004 \
  --verbose 0
fi

# league mgr
if [ $role == league_mgr ]
then
python -i -m tleague.bin.run_league_mgr \
  --port=20005 \
  --model_pool_addrs=localhost:10003:10004 \
  --game_mgr_type="${game_mgr_type}" \
  --game_mgr_config="${game_mgr_config}" \
  --mutable_hyperparam_type="${mutable_hyperparam_type}" \
  --hyperparam_config_name="${hyperparam_config_name}" \
  --restore_checkpoint_dir="" \
  --init_model_paths="[]" \
  --save_checkpoint_root=./tmp-trvd-yymmdd_chkpoints \
  --save_interval_secs=85 \
  --mute_actor_msg \
  --pseudo_learner_num=-1 \
  --verbose=0
fi

# learner
if [ $role == learner ]
then
python -i -m lifelike.bin.run_pg_learner \
  --learner_spec=0:30003:30004 \
  --model_pool_addrs=localhost:10003:10004 \
  --league_mgr_addr=localhost:20005 \
  --learner_id=lrngrp0 \
  --unroll_length=128 \
  --rollout_length=8 \
  --batch_size=256 \
  --rm_size=1024 \
  --pub_interval=5 \
  --log_interval=4 \
  --total_timesteps=20000000000000 \
  --burn_in_timesteps=12 \
  --outer_env="${outer_env_2}" \
  --env_config="${env_config}" \
  --policy="${policy}" \
  --policy_config="${learner_policy_config}" \
  --batch_worker_num=1 \
  --norwd_shape \
  --learner_config="${learner_config}" \
  --type=PPO
fi

#--env="${env}" \

# actor
if [ $role == actor ]
then
python -i -m lifelike.bin.run_pg_actor \
  --model_pool_addrs=localhost:10003:10004 \
  --league_mgr_addr=localhost:20005 \
  --learner_addr=localhost:30003:30004 \
  --unroll_length=128 \
  --update_model_freq=320 \
  --outer_env="${outer_env}" \
  --env_config="${env_config}" \
  --interface_config="${interface_config}" \
  --policy="${policy}" \
  --policy_config="${actor_policy_config}" \
  --log_interval_steps=3 \
  --n_v=1 \
  --rwd_shape \
  --nodistillation \
  --verbose=0 \
  --type="${actor_type}"
fi
