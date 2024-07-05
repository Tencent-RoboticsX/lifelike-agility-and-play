from collections import OrderedDict

import lifelike.networks.layers as tair_layers
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import tensorflow.contrib.layers as tfc_layers
import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
import tpolicies.tp_utils as tp_utils
from lifelike.networks.legged_robot.epmc_net.epmc_net_data import \
    EPMCInputs, EPMCOutputs, EPMCTrainableVariables, \
    EMPCConfig, EPMCLosses
from lifelike.networks.legged_robot.pmc_net.pmc_net import llc, _normc_initializer
from lifelike.networks.legged_robot.pmc_net.pmc_net import llc as llc_light
from tensorflow.contrib.framework import nest
from tpolicies.utils.distributions import DiagGaussianPdType, CategoricalPdType
from tpolicies.utils.sequence_ops import multistep_forward_view
from tpolicies.utils.distributions import make_pdtype
import numpy as np


def _make_vars(scope) -> EPMCTrainableVariables:
    scope = scope if isinstance(scope, str) else scope.name + '/'
    all_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES, scope)
    vf_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES,
                                 '{}.*{}'.format(scope, 'vf'))
    pf_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES,
                                 '{}.*{}'.format(scope, 'pol'))
    ob_stat = tfc.get_collection(tfc.GraphKeys.GLOBAL_VARIABLES,
                                 '{}.*{}'.format(scope, 'obfilter'))
    lstm_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES,
                                   '{}.*{}'.format(scope, 'lstm_embed'))
    return EPMCTrainableVariables(
        all_vars=all_vars, vf_vars=vf_vars, pf_vars=pf_vars, ob_stat=ob_stat, lstm_vars=lstm_vars)


def epmc_inputs_placeholder(nc: EMPCConfig):
    """create the inputs placeholder for MLP"""
    x_ph = tp_utils.placeholders_from_gym_space(
        nc.ob_space, batch_size=nc.batch_size, name='ob_ph')

    if nc.test:
        # when testing, there are no ground-truth actions
        a_ph = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
    else:
        a_ph = tp_utils.placeholders_from_gym_space(
            nc.ac_space, batch_size=nc.batch_size, name='ac_ph')

    neglogp = tp_utils.map_gym_space_to_structure(
        func=lambda x_sp: tf.placeholder(shape=(nc.batch_size,),
                                         dtype=tf.float32,
                                         name='neglogp'),
        gym_sp=nc.ac_space
    )

    n_v = nc.n_v  # no. of value heads
    discount = tf.placeholder(tf.float32, (nc.batch_size,), 'discount')
    r = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'r')
    ret = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'R')
    value = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'V')
    S = tf.placeholder(tf.float32, (nc.batch_size, nc.hs_len), 'hs')
    M = tf.placeholder(tf.float32, (nc.batch_size,), 'hsm')
    flatparam = tp_utils.map_gym_space_to_structure(
        func=lambda x_sp: tf.placeholder(shape=(nc.batch_size,) + tuple(make_pdtype(x_sp).param_shape()),
                                         dtype=np.float32,),
        gym_sp=nc.ac_space
    )
    return EPMCInputs(
        X=x_ph,
        A=a_ph,
        neglogp=neglogp,
        discount=discount,
        r=r,
        R=ret,
        V=value,
        S=S,
        M=M,
        flatparam=flatparam,
    )


def percep_2d_encoder(x_2d, scope='percep_2d'):
    with tfc.variable_scope(scope, reuse=tf.AUTO_REUSE):
        embed = tf.expand_dims(x_2d, axis=-1)  # [B, H, W, C]
        embed = tfc_layers.conv2d(embed, 4, [1, 1])
        embed = tfc_layers.conv2d(embed, 4, [4, 4], stride=2)
        embed = tfc_layers.conv2d(embed, 4, [2, 2], stride=2)
        embed = tfc_layers.conv2d(embed, 1, [2, 2])
        embed = tfc_layers.flatten(embed)
        return embed


def periodic_padding_1d(x, padding=1):
    '''
    x: shape (batch_size, width)
    return x padded with periodic boundaries. i.e. torus or donut
    '''
    p = padding
    pad_left = x[:, -p:]
    pad_right = x[:, :p]
    padded_x = tf.concat([pad_left, x, pad_right], axis=-1)
    return padded_x


def percep_1d_encoder(x_1d, nc, kernel_size=4):
    with tfc.variable_scope('percep_1d', reuse=tf.AUTO_REUSE):
        padded_x_1d = periodic_padding_1d(x_1d, padding=kernel_size)
        embed = tf.expand_dims(padded_x_1d, axis=-1)  # [B, num of words, word embed dim]
        embed = tfc_layers.conv1d(embed, 4, kernel_size, padding='SAME')  # tfc_layers.conv1d does not support CIRCULAR
        embed = embed[:, kernel_size:-kernel_size, :]
        embed = tfc_layers.conv1d(embed, 4, kernel_size, stride=2)
        embed = tfc_layers.conv1d(embed, 4, kernel_size, stride=2)
        embed = tfc_layers.conv1d(embed, 1, kernel_size)
        return tf.reshape(embed, shape=(nc.batch_size, -1))


def usr_cmd_encoder(usr_cmd, nc):
    usr_cmd_all_embeds = []
    percep_2d_embed = percep_2d_encoder(usr_cmd['percep_2d'], scope='percep_2d')
    usr_cmd_all_embeds.append(percep_2d_embed)
    percep_1d_embed = percep_1d_encoder(usr_cmd['percep_1d'], nc)
    usr_cmd_all_embeds.append(percep_1d_embed)
    percep_front_embed = percep_2d_encoder(usr_cmd['percep_front'], scope='percep_front')
    usr_cmd_all_embeds.append(percep_front_embed)
    percep_vec_embed = tfc_layers.fully_connected(usr_cmd['target'], 32,
                                                  activation_fn=nc.main_activation_func_op)
    usr_cmd_all_embeds = [percep_vec_embed] + usr_cmd_all_embeds
    usr_cmd_embed = tf.concat(usr_cmd_all_embeds, axis=-1)
    usr_cmd_embed = tfc_layers.fully_connected(usr_cmd_embed, nc.bot_neck_prop_embed_size,
                                               activation_fn=nc.main_activation_func_op)
    return usr_cmd_embed


def mlc_encoder(prop, usr_cmd, hs, m, nc):
    with tfc.variable_scope('mlc_encoder', reuse=tf.AUTO_REUSE):
        prop_embed = tfc_layers.fully_connected(prop, nc.bot_neck_prop_embed_size,
                                                activation_fn=nc.main_activation_func_op)
        usr_cmd_embed = usr_cmd_encoder(usr_cmd, nc)
        embed = tf.concat([prop_embed, usr_cmd_embed], axis=-1)
        embed = tfc_layers.fully_connected(embed, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        if nc.expert_lstm:
            lstm_embed, hs_new = tp_layers.lstm_embed_block(
                inputs_x=embed,
                inputs_hs=hs,
                inputs_mask=m,
                nc=nc)
        else:
            lstm_embed = tfc_layers.fully_connected(embed, nc.embed_dim, activation_fn=nc.main_activation_func_op)
            hs_new = tf.zeros(shape=(nc.nrollout, nc.hs_len))
        if nc.discrete_z:
            z_logits = tfc_layers.fully_connected(lstm_embed, nc.z_len, activation_fn=None)
            z_head = tp_layers.to_action_head(z_logits, CategoricalPdType)
        else:
            z_mu = tfc_layers.fully_connected(lstm_embed, nc.z_len, activation_fn=None)
            if nc.isolate_z_logvar:
                z_logvar = tf.get_variable(name='logvar', shape=(1, nc.z_len),
                                           initializer=tf.constant_initializer(nc.z_logvar_init))
                z_logvar = tf.tile(z_logvar, [nc.batch_size, 1])
            else:
                z_logvar = tfc_layers.fully_connected(lstm_embed, nc.z_len, activation_fn=None)
            pdparams = tf.concat([z_mu, z_mu * 0.0 + z_logvar], axis=1)
            z_head = tp_layers.to_action_head(pdparams, DiagGaussianPdType)
    return z_head, hs_new


def mapping_z(encoding_indices, z_len, num_embeddings):
    #  https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
    codebook = tf.get_variable(name='embedding', shape=(z_len, num_embeddings),
                               initializer=tf.uniform_unit_scaling_initializer, trainable=True)
    encodings = tf.one_hot(encoding_indices, num_embeddings)
    with tf.control_dependencies([encoding_indices]):
        w_trans = tf.transpose(codebook.read_value(), [1, 0])
        quantized = tf.nn.embedding_lookup(w_trans, encoding_indices, validate_indices=False)
    return quantized, encodings


def epmc_net(inputs: EPMCInputs,
             nc: EMPCConfig,
             scope=None) -> EPMCOutputs:
    """create the whole net for CutTrackingZ"""
    with tfc.variable_scope(scope, default_name='model') as sc:
        # lstm related
        # nc_lstm = deepcopy(nc) # deepcopy raises Error when using multi-GPU & hvd
        assert nc.hs_len % 3 == 0, 'Use separate pi and vf lstm networks, and z'
        nc.hs_len = nc.hs_len // 3

        # input
        prop = inputs.X['prop']
        if nc.append_hist_a:
            prop = tf.concat([prop, inputs.X['prop_a']], axis=-1)
        usr_cmd_vf = OrderedDict({
            'percep_2d': inputs.X['percep_2d'],
            'percep_1d': inputs.X['percep_1d'],
            'percep_front': inputs.X['percep_front'],
        })
        usr_cmd_pi = OrderedDict({
            'percep_2d': inputs.X['percep_2d'],
            'percep_1d': inputs.X['percep_1d'],
            'percep_front': inputs.X['percep_front'],
        })
        usr_cmd_vf.update({
            'target': inputs.X['target'],
        })
        usr_cmd_pi.update({
            'target': inputs.X['target'],
        })

        # Fixed order of hs
        hs_vf = inputs.S[:, :nc.hs_len]
        hs_pi = inputs.S[:, nc.hs_len:nc.hs_len * 2]
        hs_z = inputs.S[:, nc.hs_len * 2:]

        # obs normalization
        if nc.rms_momentum is not None:
            with tfc.variable_scope(nc.llc_param_type + '/rms'):
                ob, _ = tair_layers.rms(prop, momentum=nc.rms_momentum)
                ob_rms = tf.clip_by_value(tf.stop_gradient(ob), -5.0, 5.0)
                prop_rms = ob_rms
        else:
            prop_rms = ob_rms = prop

        # additional perception
        prop_rms_extended = prop_rms

        # value
        if nc.use_value_head:
            with tfc.variable_scope('vf'):
                last_out_vf1 = tfc.nn.tanh(tfc.layers.dense(prop_rms_extended, nc.embed_dim // 2, name="fc1",
                                                            kernel_initializer=_normc_initializer(1.0)))
                usr_cmd_vf = usr_cmd_encoder(usr_cmd_vf, nc)
                last_out_vf2 = tfc.nn.tanh(tfc.layers.dense(usr_cmd_vf, nc.embed_dim // 2, name="fc2",
                                                            kernel_initializer=_normc_initializer(1.0)))
                last_out_vf = tf.concat([last_out_vf1, last_out_vf2], axis=-1)
                last_out_vf = tfc.nn.tanh(tfc.layers.dense(last_out_vf, nc.embed_dim, name="fc3",
                                                           kernel_initializer=_normc_initializer(1.0)))
                lstm_embed_vf, hs_vf_new = tp_layers.lstm_embed_block(
                    inputs_x=last_out_vf,
                    inputs_hs=hs_vf,
                    inputs_mask=inputs.M,
                    nc=nc)
                vf = tfc.layers.dense(lstm_embed_vf, nc.n_v, name='value', kernel_initializer=_normc_initializer(1.0))
        else:
            vf = 0
            hs_vf_new = hs_vf

        self_fed_heads, outer_fed_heads = None, None
        with tfc.variable_scope('expert_pi'):
            if nc.use_self_fed_heads:
                z_head, hs_z_new = mlc_encoder(prop_rms_extended, usr_cmd_pi, hs_z, inputs.M, nc)
                z_curr = z_head.sam
                with tfc.variable_scope(nc.llc_param_type):
                    if nc.discrete_z:
                        with tfc.variable_scope('llc'):
                            z_curr, encodings = mapping_z(z_curr, z_len=nc.z_len_llc, num_embeddings=nc.z_len)
                    if nc.llc_light:
                        llc_head, hs_pi_new = llc_light(prop_rms, z_curr, nc), tf.zeros_like(hs_vf_new)
                    else:
                        llc_head, hs_pi_new = llc(prop_rms, z_curr, hs_pi, inputs, nc)
                self_fed_heads = tp_utils.pack_sequence_as_structure_like_gym_space(nc.ac_space, [z_head, llc_head])
            else:
                z_head, hs_z_new = mlc_encoder(prop_rms_extended, usr_cmd_pi, hs_z, inputs.M, nc)
                flag = inputs.A['A_Z'] is not None
                assert flag, ('creating outer_fed_heads, '
                              'but outer fed heads are None ...')
                z_curr = inputs.A['A_Z']
                with tfc.variable_scope(nc.llc_param_type):
                    if nc.discrete_z:
                        with tfc.variable_scope('llc'):
                            z_curr, encodings = mapping_z(z_curr, z_len=nc.z_len_llc, num_embeddings=nc.z_len)
                    if nc.llc_light:
                        llc_head, hs_pi_new = llc_light(prop_rms, z_curr, nc), tf.zeros_like(hs_vf_new)
                    else:
                        llc_head, hs_pi_new = llc(prop_rms, z_curr, hs_pi, inputs, nc)
                outer_fed_heads = tp_utils.pack_sequence_as_structure_like_gym_space(nc.ac_space, [z_head, llc_head])

        hs_new = tf.concat([hs_vf_new, hs_pi_new, hs_z_new], axis=-1)

        # make loss
        loss = None
        with tf.variable_scope('losses'):
            # regularization loss
            total_reg_loss = tfc.losses.get_regularization_losses(scope=sc.name)
            if nc.use_loss_type in ['rl', 'ppo', 'ppo2']:
                # ppo loss
                example_ac_sp = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
                outer_fed_head_neglogp = nest.map_structure_up_to(
                    example_ac_sp,
                    lambda head, ac: head.pd.neglogp(ac),
                    outer_fed_heads,
                    inputs.A)

                if nc.use_loss_type in ['rl', 'ppo']:
                    pg_loss, value_loss = tp_losses.ppo_loss(
                        neglogp=outer_fed_head_neglogp,
                        oldneglogp=inputs.neglogp,
                        vpred=vf,
                        R=inputs.R,
                        V=inputs.V,
                        masks=None,
                        reward_weights=None,
                        adv_normalize=True,
                        sync_statistics=nc.sync_statistics,
                        clip_range_lower=nc.clip_range_lower,
                    )
                    pg_loss = tf.reduce_mean(pg_loss)
                    value_loss = tf.reduce_mean(value_loss)
                elif nc.use_loss_type == 'ppo2':
                    def _batch_to_tb(tsr):
                        return tf.transpose(tf.reshape(
                            tsr, shape=(nc.nrollout, nc.rollout_len)))

                    neglogp_list = [_batch_to_tb(neglogp)
                                    for neglogp in nest.flatten(outer_fed_head_neglogp)]
                    oldneglogp_list = [_batch_to_tb(oldneglogp)
                                       for oldneglogp in nest.flatten(inputs.neglogp)]

                    vpred_list = [_batch_to_tb(v) for v in tf.split(vf, nc.n_v, axis=1)]
                    reward_list = [_batch_to_tb(r) for r in tf.split(inputs.r, nc.n_v, axis=1)]
                    if nc.n_v > 1:
                        # # batch norm reward (equivalent to norm adv)
                        # batch_r_mean = tf.reduce_mean(inputs.r, axis=0, keepdims=True)
                        # batch_r_mean_square = tf.reduce_mean(tf.square(inputs.r - batch_r_mean),
                        #                                      axis=0, keepdims=True)
                        # if nc.sync_statistics == 'horovod':
                        #     import horovod.tensorflow as hvd
                        #     batch_r_mean = hvd.allreduce(batch_r_mean, average=True)
                        #     batch_r_mean_square = hvd.allreduce(batch_r_mean_square, average=True)
                        # r_norm = inputs.r - batch_r_mean
                        # r_norm = r_norm / tf.sqrt(batch_r_mean_square + 1e-8)
                        # # assume the last is gail reward
                        # r_norm = tf.split(inputs.r, nc.n_v, axis=1)[:-1] + tf.split(r_norm, nc.n_v, axis=1)[-1:]
                        # reward_list = [_batch_to_tb(r) for r in r_norm]
                        # reward_weights size should be consistent with n_v
                        reward_weights = tf.squeeze(tf.convert_to_tensor(nc.reward_weights, tf.float32))
                        assert reward_weights.shape.as_list()[0] == len(reward_list), (
                            'For ppo2 loss, reward_weight size must be the same with number of'
                            ' value head: each reward_weight element must correspond to one '
                            'value-head exactly.'
                        )
                    else:
                        reward_weights = 1.0
                    discounts = _batch_to_tb(inputs.discount)

                    # lambda for td-lambda or lambda-return
                    assert nc.lam is not None, 'building rl_ppo2, but lam for lambda-return is None.'
                    lam = tf.convert_to_tensor(nc.lam, tf.float32)

                    # for each value-head, compute the corresponding policy gradient loss
                    # and the value loss
                    pg_loss, value_loss = [], []
                    for vpred, reward in zip(vpred_list, reward_list):
                        """ loop over multiple value heads """
                        # compute the lambda-Return `R` in shape (T - 1, B)
                        # [:-1] means discarding the last one,
                        # [1:] means an off-one alignment.
                        # back_prop=False means R = tf.stop_gradient(R)
                        with tf.device("/cpu:0"):
                            R = multistep_forward_view(reward[:-1], discounts[:-1], vpred[1:],
                                                       lambda_=lam, back_prop=False)
                        # compute the ppo2 loss using this value-head for each of the
                        # n_action_heads action-head; then reduce them
                        # [:-1] means discarding the last one and using only T - 1 time steps
                        pg_loss_per_vh = [
                            tp_losses.ppo2_loss(
                                neglogp=neglogp[:-1],
                                oldneglogp=oldneglogp[:-1],
                                vpred=tf.stop_gradient(vpred)[:-1],
                                R=R,  # has been stop_gradient above; note in ppo_loss,
                                # R is computed in actor and naturally stop_gradient
                                mask=None,
                                adv_normalize=True,
                                clip_range=nc.clip_range,
                                clip_range_lower=nc.clip_range_lower,
                                sync_statistics=nc.sync_statistics)
                            for neglogp, oldneglogp in zip(
                                neglogp_list, oldneglogp_list)
                        ]
                        pg_loss.append(tf.reduce_sum(pg_loss_per_vh))
                        # compute the value loss for this value-head
                        value_head_loss = tf.reduce_mean(0.5 * tf.square(R - vpred[:-1]))
                        value_loss.append(value_head_loss)
                    # normally pg_loss has two dims [n_head, n_v], absorbing with merge_pi and reward_weights
                    pg_loss = tf.reduce_sum(tf.stack(pg_loss) * reward_weights)
                    value_loss = tf.reduce_sum(tf.stack(value_loss) * reward_weights)
                else:
                    raise NotImplementedError('Unknown loss type.')

                # distill loss
                distill_loss = None
                if nc.distillation:
                    distill_loss = tf.zeros(shape=())
                    outer_fed_head_pds = nest.map_structure_up_to(example_ac_sp, lambda head: head.pd, outer_fed_heads)
                    # distill_loss = tp_losses.distill_loss(
                    #     student_pds=outer_fed_head_pds,
                    #     teacher_logits=inputs.flatparam)
                    if nc.distill_llc:
                        llc_pd = outer_fed_head_pds['A_LLC']
                        llc_teacher_logit = inputs.flatparam['A_LLC']
                        llc_distill_loss = tp_losses.distill_loss(llc_pd, llc_teacher_logit)
                        distill_loss += llc_distill_loss
                    if nc.distill_z:
                        z_pd = outer_fed_head_pds['A_Z']
                        z_teacher_logit = inputs.flatparam['A_Z']
                        z_distill_loss = tp_losses.distill_loss(z_pd, z_teacher_logit)
                        distill_loss += z_distill_loss

                # entropy loss
                entropy_loss = nest.map_structure_up_to(
                    example_ac_sp, lambda head: tf.reduce_mean(head.ent), outer_fed_heads)
                loss_endpoints = {'pg_loss': tf.reduce_mean(pg_loss),
                                  'value_loss': tf.reduce_mean(value_loss),
                                  'z_head_entropy': entropy_loss['A_Z'],
                                  'llc_head_entropy': entropy_loss['A_LLC'],

                                  'return': tf.reduce_mean(tf.reduce_mean(vf) if nc.use_loss_type == 'ppo2' else
                                                           inputs.R),
                                  # 'rms_loss': tf.reduce_mean(rms_loss),
                                  }
                if nc.distillation:
                    loss_endpoints.update({
                        'total_distill_loss': distill_loss,
                    })
                    if nc.distill_llc:
                        loss_endpoints.update({
                            'llc_distill_loss': llc_distill_loss,
                        })
                    if nc.distill_z:
                        loss_endpoints.update({
                            'z_distill_loss': z_distill_loss,
                        })

                if nc.discrete_z:
                    avg_probs = tf.reduce_mean(encodings, 0)
                    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))
                    loss_endpoints.update({
                        'perplexity': tf.reduce_mean(perplexity),
                    })
                else:
                    loss_endpoints.update({
                        'z_curr': tf.reduce_mean(z_curr),
                        'mu': tf.reduce_mean(z_head.pd.mean),
                        'logvar': tf.reduce_mean(z_head.pd.logstd),
                    })

                loss = EPMCLosses(
                    total_reg_loss=total_reg_loss,
                    pg_loss=pg_loss,
                    value_loss=value_loss,
                    entropy_loss=entropy_loss,
                    distill_loss=distill_loss,
                    loss_endpoints=loss_endpoints
                )
            elif nc.use_loss_type == 'distill':
                # fake ppo loss
                pg_loss = tf.constant(0.0)
                value_loss = tf.constant(0.0)
                entropy_loss = tf.constant(0.0)

                encodings = tf.one_hot(inputs.A['A_Z'], nc.z_len)

                teacher_z_head = tp_layers.to_action_head(inputs.flatparam['A_Z'], CategoricalPdType)
                distill_loss = z_head.pd.kl(teacher_z_head.pd)

                avg_probs = tf.reduce_mean(encodings, 0)
                perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

                loss_endpoints = {'pg_loss': tf.reduce_mean(pg_loss),
                                  'value_loss': tf.reduce_mean(value_loss),
                                  'entropy_loss': tf.reduce_mean(entropy_loss),
                                  'distill_loss': tf.reduce_mean(distill_loss),
                                  'perplexity': tf.reduce_mean(perplexity),
                                  }
                loss = EPMCLosses(
                    total_reg_loss=total_reg_loss,
                    pg_loss=pg_loss,
                    value_loss=value_loss,
                    entropy_loss=entropy_loss,
                    distill_loss=distill_loss,
                    loss_endpoints=loss_endpoints
                )

        # collect vars, endpoints, etc.
        trainable_vars = _make_vars(sc)
        endpoints = OrderedDict()
        # endpoints['z_curr'] = z_curr
    return EPMCOutputs(
        self_fed_heads=self_fed_heads,
        outer_fed_heads=outer_fed_heads,
        loss=loss,
        vars=trainable_vars,
        endpoints=endpoints,
        value_head=vf,
        ob_rms=ob_rms,
        S=hs_new,
    )


# APIs
net_build_fun = epmc_net
net_config_cls = EMPCConfig
net_inputs_placeholders_fun = epmc_inputs_placeholder
