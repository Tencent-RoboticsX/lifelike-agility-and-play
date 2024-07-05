# Usage:
#   1. open four terminals;
#   2. "bash example_pmc_train.sh model_pool" in terminal 1;
#   3. "bash example_pmc_train.sh league_mgr" in terminal 2;
#   4. "bash example_pmc_train.sh actor" in terminal 3;
#   5. "bash example_pmc_train.sh learner" in terminal 4;


role=$1
# common args
actor_type=PPO
outer_env=lifelike.sim_envs.pybullet_envs.create_tracking_game
outer_env_2=lifelike.sim_envs.pybullet_envs.create_tracking_env

game_mgr_type=tleague.game_mgr.game_mgrs.SelfPlayGameMgr && \
game_mgr_config="{
  'max_n_players': 1}"
mutable_hyperparam_type=ConstantHyperparam
hyperparam_config_name="{ \
  'learning_rate': 0.00001, \
  'lam': 0.95, \
  'gamma': 0.95, \
}" && \
policy=lifelike.networks.legged_robot.pmc_net.pmc_net
learner_policy_config="{ \
  'test': False, \
  'rl': True, \
  'use_loss_type': 'rl', \
  'z_prior_type': 'VQ', \
  'use_value_head': True, \
  'rms_momentum': 0.0001, \
  'append_hist_a': True, \
  'main_activation_func': 'relu', \
  'n_v': 1, \
  'use_lstm': False, \
  'z_len': 32, \
  'num_embeddings': 256, \
  'conditional': True, \
  'bot_neck_z_embed_size': 32, \
  'bot_neck_prop_embed_size': 64, \
}" && \
actor_policy_config="{ \
  'batch_size': 1, \
  'rollout_len': 1, \
  'test': True, \
  'use_loss_type': 'none', \
  'z_prior_type': 'VQ', \
  'use_value_head': True, \
  'rms_momentum': 0.0001, \
  'append_hist_a': True, \
  'main_activation_func': 'relu', \
  'n_v': 1, \
  'use_lstm': False, \
  'z_len': 32, \
  'conditional': True, \
  'bot_neck_z_embed_size': 32, \
  'bot_neck_prop_embed_size': 64, \
  'sync_statistics': 'none', \
}" && \
learner_config="{ \
  'vf_coef': 1, \
  'max_grad_norm': 0.5, \
  'distill_coef': 0.0, \
  'ent_coef': 0.00000, \
  'ep_loss_coef': {'q_latent_loss':1.0, 'e_latent_loss':0.25, 'rms_loss': 1.0}, \
}" && \
env_config="{ \
  'arena_id': 'LeggedRobotTracking', \
  'render': False, \
  'data_path': '../data/mocap_data', \
  'control_freq': 50.0, \
  'prop_type': ['joint_pos', 'joint_vel', 'root_ang_vel_loc', 'root_lin_vel_loc', 'e_g'], \
  'prioritized_sample_factor': 3.0, \
  'set_obstacle': True, \
  'kp': 50.0, \
  'kd': 0.5, \
  'max_tau': 18, \
  'reward_weights': {'joint_pos': 0.3, 'joint_vel': 0.05, 'end_effector': 0.1, 'root_pose': 0.5, 'root_vel': 0.05,}, \
}" && \

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
  --policy="${policy}" \
  --policy_config="${actor_policy_config}" \
  --log_interval_steps=3 \
  --n_v=1 \
  --rwd_shape \
  --nodistillation \
  --verbose=0 \
  --type="${actor_type}"
fi
