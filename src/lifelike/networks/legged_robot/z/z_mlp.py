from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc

import tpolicies.tp_utils as tp_utils
import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
import tensorflow.contrib.layers as tfc_layers
from tpolicies.utils.distributions import DiagGaussianPdType

import lifelike.networks.layers as tair_layers
from lifelike.networks.legged_robot.z.z_mlp_data import ZMLPInputs, ZMLPOutputs, \
    ZMLPTrainableVariables, ZMLPLosses, ZMLPConfig
from lifelike.networks.utils import _normc_initializer


def _make_vars(scope) -> ZMLPTrainableVariables:
    scope = scope if isinstance(scope, str) else scope.name + '/'
    all_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES, scope)
    encoder_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES,
                                      '{}.*{}'.format(scope, 'encoder'))
    decoder_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES,
                                      '{}.*{}'.format(scope, 'decoder'))
    lstm_vars = tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES,
                                   '{}.*{}'.format(scope, 'lstm'))
    return ZMLPTrainableVariables(all_vars=all_vars,
                                  encoder_vars=encoder_vars,
                                  decoder_vars=decoder_vars,
                                  lstm_vars=lstm_vars)


def z_mlp_inputs_placeholder(nc: ZMLPConfig):
    """create the inputs placeholder for MLP"""
    x_ph = tp_utils.placeholders_from_gym_space(
        nc.ob_space, batch_size=nc.batch_size, name='ob_ph')

    # if nc.test:
    #     # when testing, there are no ground-truth actions
    #     a_ph = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
    # else:
    a_ph = tp_utils.placeholders_from_gym_space(
        nc.ac_space, batch_size=nc.batch_size, name='ac_ph')

    neglogp = tp_utils.map_gym_space_to_structure(
        func=lambda x_sp: tf.placeholder(shape=(nc.batch_size,),
                                         dtype=tf.float32,
                                         name='neglogp'),
        gym_sp=nc.ac_space
    )

    n_v = 1  # no. of value heads
    ret = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'R')
    value = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'V')
    S = tf.placeholder(tf.float32, (nc.batch_size, nc.z_len), 'z')
    M = tf.placeholder(tf.float32, (nc.batch_size,), 'hsm')
    flatparam = tf.placeholder(tf.float32, (nc.batch_size, nc.ac_space.shape[0] * 2), 'hsm')

    return ZMLPInputs(
        X=x_ph,
        A=a_ph,
        neglogp=neglogp,
        R=ret,
        V=value,
        S=S,
        M=M,
        flatparam=flatparam,
    )


def mlp_encoder(x, nc):
    with tfc.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        embed = tfc_layers.fully_connected(x, nc.enc_dim, activation_fn=tf.nn.relu)
        embed = tfc_layers.fully_connected(embed, nc.enc_dim, activation_fn=tf.nn.relu)
        out_mean = tfc_layers.fully_connected(embed, nc.z_len, activation_fn=None)
        out_logvar = tfc_layers.fully_connected(embed, nc.z_len, activation_fn=None)
    return out_mean, out_logvar


def mlp_decoder(x, nc):
    with tfc.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        embed = tfc_layers.fully_connected(x, nc.dec_dim, activation_fn=tf.nn.relu)
        embed = tfc_layers.fully_connected(embed, nc.dec_dim, activation_fn=tf.nn.relu)
        embed = tfc_layers.fully_connected(embed, nc.dec_dim, activation_fn=tf.nn.relu)
        out = tfc_layers.fully_connected(embed, nc.ac_space.shape[0],
                                         activation_fn=None,
                                         weights_initializer=_normc_initializer(0.01),
                                         scope='mean')
    return out


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def z_mlp(inputs: ZMLPInputs,
          nc: ZMLPConfig,
          scope=None) -> ZMLPOutputs:
    """ create the whole net """
    with tfc.variable_scope(scope, default_name='z_mlp') as sc:
        ob = inputs.X
        # For supervised learning or distillation, only use the first z from data
        z_prev = tf.reshape(inputs.S, shape=[nc.nrollout, nc.rollout_len, -1])[:, 0, :]
        masks = tf.reshape(inputs.M, shape=[nc.nrollout, nc.rollout_len])

        with tfc.variable_scope('rms'):
            ob, rms_loss = tair_layers.rms(inputs=ob, momentum=nc.rms_momentum)
            ob_rms = obz = tf.stop_gradient(tf.clip_by_value(ob, -5.0, 5.0))

        with tfc.variable_scope('pi'):
            obz_seq = tf.reshape(obz, shape=[nc.nrollout, nc.rollout_len, -1])  # [rollout_len, nrollout, dim]
            prop = obz_seq[:, :, 0:nc.prop_dim]
            future = obz_seq[:, :, nc.prop_dim:]

            logstd = tf.get_variable(name='logstd', shape=[1, nc.ac_space.shape[0]],
                                     initializer=tf.constant_initializer(nc.logstd_init))

            zs = []
            zs_prev = []
            mus = []
            logvars = []
            heads = []
            for t in range(nc.rollout_len):
                prop_t = prop[:, t, :]
                future_t = future[:, t, :]
                # z encoder
                # Carefully notifying mask below
                s_enc = tf.concat([future_t, z_prev * (1 - tf.cast(masks[:, t:t+1], tf.float32))], axis=-1)
                mu, logvar = mlp_encoder(s_enc, nc)
                z_curr = reparameterize(mu, logvar)
                # low-lower controller
                s_dec = tf.concat([prop_t, z_curr], axis=-1)
                mean = mlp_decoder(s_dec, nc)
                pdparams = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                head = tp_layers.to_action_head(pdparams, DiagGaussianPdType)

                zs_prev.append(z_prev)
                z_prev = z_curr
                zs.append(z_curr)
                mus.append(mu)
                logvars.append(logvar)
                heads.append(head)

        hs_new = zs[-1]

        # make loss
        loss = None
        if nc.use_loss_type == 'distill':
            # regularization loss
            total_reg_loss = tfc.losses.get_regularization_losses(scope=sc.name)
            with tf.variable_scope('losses'):
                # fake ppo loss
                pg_loss = tf.constant(0.0)
                value_loss = tf.constant(0.0)
                entropy_loss = tf.constant(0.0)

                distill_loss = 0
                logpz = 0
                logqz_x = 0
                for t in range(nc.rollout_len):
                    if nc.distill_loss_type == 'standard':
                        flatparam = inputs.flatparam
                        flatparam_seq = tf.reshape(flatparam, shape=[nc.nrollout, nc.rollout_len, -1])
                        distill_loss += tp_losses.distill_loss_v2(
                            student_pds=heads[t].pd,
                            teacher_mean=tf.split(flatparam_seq[:, t, :], axis=-1, num_or_size_splits=2)[0],
                            teacher_logstd=tf.split(flatparam_seq[:, t, :], axis=-1, num_or_size_splits=2)[1],
                            masks=True)
                    elif nc.distill_loss_type == 'action_var':
                        a_seq = tf.reshape(inputs.A, shape=[nc.nrollout, nc.rollout_len, -1])
                        distill_loss += tp_losses.distill_loss_v2(
                            student_pds=heads[t].pd,
                            teacher_mean=a_seq[:, t, :],
                            teacher_logstd=tf.ones_like(a_seq[:, t, :]) * nc.action_var,
                            masks=True)
                    elif nc.distill_loss_type == 'supervised':
                        a_seq = tf.reshape(inputs.A, shape=[nc.nrollout, nc.rollout_len, -1])
                        distill_loss += tf.reduce_sum((heads[t].pd.mean - a_seq[:, t, :]) ** 2, axis=-1)
                    # Carefully notifying mask below
                    logpz += log_normal_pdf(zs[t], nc.alpha * zs_prev[t] * tf.cast(masks[:, t:t+1], tf.float32),
                                            nc.logvar_prior)
                    logqz_x += log_normal_pdf(zs[t], mus[t], logvars[t])
                distill_loss -= nc.beta * (logpz - logqz_x)
                distill_loss /= nc.rollout_len

                loss_endpoints = {'pg_loss': tf.reduce_mean(pg_loss),
                                  'value_loss': tf.reduce_mean(value_loss),
                                  'entropy_loss': tf.reduce_mean(entropy_loss),
                                  'return': tf.reduce_mean(inputs.R) if nc.use_loss_type == 'rl' else tf.constant(0.0),
                                  'rms_loss': tf.reduce_mean(rms_loss),
                                  'distill_loss': tf.reduce_mean(distill_loss)}

                loss = ZMLPLosses(
                    total_reg_loss=total_reg_loss,
                    pg_loss=pg_loss,
                    value_loss=value_loss,
                    entropy_loss=entropy_loss,
                    loss_endpoints=loss_endpoints
                )

        # collect vars, endpoints, etc.
        trainable_vars = _make_vars(sc)
        endpoints = OrderedDict()  # add something wanted

    return ZMLPOutputs(
        self_fed_heads=head,
        outer_fed_heads=head,
        loss=loss,
        vars=trainable_vars,
        endpoints=endpoints,
        value_head=None,
        ob_rms=ob_rms,
        S=hs_new,
    )


# APIs
net_build_fun = z_mlp
net_config_cls = ZMLPConfig
net_inputs_placeholders_fun = z_mlp_inputs_placeholder
