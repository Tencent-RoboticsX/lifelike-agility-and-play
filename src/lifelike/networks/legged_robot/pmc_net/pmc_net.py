from collections import OrderedDict

import lifelike.networks.layers as tair_layers
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import tensorflow.contrib.layers as tfc_layers
import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
import tpolicies.tp_utils as tp_utils
from lifelike.networks.legged_robot.pmc_net.pmc_net_data import PMCInputs, \
    PMCOutputs, PMCTrainableVariables, PMCLosses, PMCConfig
from lifelike.networks.legged_robot.z.z_mlp import reparameterize, log_normal_pdf
from lifelike.networks.utils import _normc_initializer
from tpolicies.utils.distributions import DiagGaussianPdType
from tpolicies.utils.sequence_ops import multistep_forward_view


def _make_vars(scope) -> PMCTrainableVariables:
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
    return PMCTrainableVariables(all_vars=all_vars, vf_vars=vf_vars, pf_vars=pf_vars, ob_stat=ob_stat,
                                 lstm_vars=lstm_vars)


def encoder(x, nc):
    with tfc.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        embed = tfc_layers.fully_connected(x, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        embed = tfc_layers.fully_connected(embed, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        out_mean = tfc_layers.fully_connected(embed, nc.z_len, activation_fn=None)
        out_logvar = tfc_layers.fully_connected(embed, nc.z_len, activation_fn=None)
    return out_mean, out_logvar


def vq_encoder(x, nc):
    with tfc.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        embed = tfc_layers.fully_connected(x, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        embed = tfc_layers.fully_connected(embed, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        z = tfc_layers.fully_connected(embed, nc.z_len, activation_fn=None)
    return z


def decoder(x, nc):
    with tfc.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        embed = tfc_layers.fully_connected(x, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        embed = tfc_layers.fully_connected(embed, nc.embed_dim, activation_fn=nc.main_activation_func_op)
        out = tfc_layers.fully_connected(embed, 12,
                                         activation_fn=None,
                                         weights_initializer=_normc_initializer(0.01),
                                         scope='mean')
    return out


def pmc_inputs_placeholder(nc: PMCConfig):
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

    n_v = 1  # no. of value heads
    discount = tf.placeholder(tf.float32, (nc.batch_size,), 'discount')
    r = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'r')
    ret = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'R')
    value = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'V')
    flatparam = tf.placeholder(tf.float32, (nc.batch_size, nc.ac_space.shape[0] * 2), 'hsm')

    return PMCInputs(
        X=x_ph,
        A=a_ph,
        neglogp=neglogp,
        discount=discount,
        r=r,
        R=ret,
        V=value,
        flatparam=flatparam,
    )


def llc(prop_rms, z_curr, nc):
    with tfc.variable_scope('llc'):
        # separate embed
        prop_embed = tfc_layers.fully_connected(prop_rms, nc.bot_neck_prop_embed_size,
                                                activation_fn=nc.main_activation_func_op)
        z_curr_embed = tfc_layers.fully_connected(z_curr, nc.bot_neck_z_embed_size,
                                                  activation_fn=nc.main_activation_func_op)
        # low-lower controller
        s_dec = tf.concat([prop_embed, z_curr_embed], axis=-1)
        mean = decoder(s_dec, nc)
        logstd = tf.get_variable(name='logstd', shape=(1, 12),
                                 initializer=tf.constant_initializer(nc.logstd_init))

        pdparams = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        head = tp_layers.to_action_head(pdparams, DiagGaussianPdType)
    return head


def pmc_net(inputs: PMCInputs,
            nc: PMCConfig,
            scope=None) -> PMCOutputs:
    """create the whole net for TrackingZ"""
    with tfc.variable_scope(scope, default_name='model') as sc:
        # input
        prop = inputs.X['prop']
        if nc.append_hist_a:
            prop = tf.concat([prop, inputs.X['prop_a']], axis=-1)
        future = inputs.X['future']

        # obs normalization
        if nc.rms_momentum is not None:
            with tfc.variable_scope('prop_rms'):
                prop_rms, prop_rms_loss = tair_layers.rms(inputs=prop, momentum=nc.rms_momentum)
                prop_rms = tf.clip_by_value(tf.stop_gradient(prop_rms), -5.0, 5.0)
            with tfc.variable_scope('future_rms'):
                future_rms, future_rms_loss = tair_layers.rms(future, momentum=nc.rms_momentum)
                future_rms = tf.clip_by_value(tf.stop_gradient(future_rms), -5.0, 5.0)
            rms_loss = tf.reduce_mean(prop_rms_loss) + tf.reduce_mean(future_rms_loss)
        else:
            prop_rms, future_rms = prop, future
        ob_rms = tf.concat([prop_rms, future_rms], axis=-1)

        with tfc.variable_scope('vf'):
            last_out_vf = tfc.nn.tanh(tfc.layers.dense(ob_rms, nc.embed_dim, name="fc1",
                                                       kernel_initializer=_normc_initializer(1.0)))
            last_out_vf = tfc.nn.tanh(tfc.layers.dense(last_out_vf, nc.embed_dim, name="fc2",
                                                       kernel_initializer=_normc_initializer(1.0)))
            vf = tfc.layers.dense(last_out_vf, 1, name='value', kernel_initializer=_normc_initializer(1.0))

        with tfc.variable_scope('pi'):
            # z encoder
            if nc.z_prior_type == 'Gaussian':
                if nc.conditional:
                    mu, logvar = encoder(ob_rms, nc)
                else:
                    mu, logvar = encoder(future_rms, nc)
                z_curr = reparameterize(mu, logvar)
            elif nc.z_prior_type == 'VQ':
                z_encode = vq_encoder(ob_rms, nc)
                #  https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
                with tfc.variable_scope('llc'):
                    codebook = tf.get_variable(name='embedding', shape=(nc.z_len, nc.num_embeddings),
                                               initializer=tf.uniform_unit_scaling_initializer, trainable=True)
                # Assert last dimension is same as nc.embedding_dim
                flat_inputs = tf.reshape(z_encode, [-1, nc.z_len])
                distances = (tf.reduce_sum(flat_inputs ** 2, 1, keepdims=True)
                             - 2 * tf.matmul(flat_inputs, codebook)
                             + tf.reduce_sum(codebook ** 2, 0, keepdims=True))
                encoding_indices = tf.argmax(-distances, 1)
                """ use the following line to test uniform random z; note VQ-VAE requires learning the prior of z
                 and the prior should not be uniform """
                # encoding_indices = tf.random.uniform(shape=encoding_indices.shape, minval=0,
                #                                      maxval=nc.num_embeddings-1, dtype=tf.dtypes.int32)
                encodings = tf.one_hot(encoding_indices, nc.num_embeddings)
                encoding_indices = tf.reshape(encoding_indices, tf.shape(z_encode)[:-1])
                with tf.control_dependencies([encoding_indices]):
                    w_trans = tf.transpose(codebook.read_value(), [1, 0])
                    quantized = tf.nn.embedding_lookup(w_trans, encoding_indices, validate_indices=False)
                z_curr = z_encode + tf.stop_gradient(quantized - z_encode)
            head = llc(prop_rms, z_curr, nc)

        endpoints = OrderedDict()  # TODO
        # make loss
        loss = None
        with tf.variable_scope('losses'):
            # regularization loss
            total_reg_loss = tfc.losses.get_regularization_losses(scope=sc.name)
            if nc.use_loss_type in ['rl', 'ppo', 'ppo2']:
                # ppo loss
                if nc.use_loss_type in ['rl', 'ppo']:
                    neglogp = head.pd.neglogp(inputs.A)
                    pg_loss, value_loss = tp_losses.ppo_loss(
                        neglogp=neglogp,
                        oldneglogp=inputs.neglogp,
                        vpred=vf,
                        R=inputs.R,
                        V=inputs.V,
                        masks=None,
                        reward_weights=None,
                        adv_normalize=True,
                        clip_range=nc.clip_range,
                        clip_range_lower=nc.clip_range_lower,
                        sync_statistics=nc.sync_statistics
                    )
                elif nc.use_loss_type == 'ppo2':
                    def _batch_to_tb(tsr):
                        # shape (batch_size, ...) -> shape (T, B, ...)
                        return tf.transpose(tf.reshape(tsr, shape=(nc.nrollout, nc.rollout_len)))

                    neglogp = _batch_to_tb(head.pd.neglogp(inputs.A))
                    oldneglogp = _batch_to_tb(inputs.neglogp)
                    vpred = _batch_to_tb(vf)
                    reward = _batch_to_tb(inputs.r)
                    discounts = _batch_to_tb(inputs.discount)

                    # lambda for td-lambda or lambda-return
                    assert nc.lam is not None, 'building rl_ppo2, but lam for lambda-return is None.'
                    lam = tf.convert_to_tensor(nc.lam, tf.float32)

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
                    pg_loss = tp_losses.ppo2_loss(
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
                    # compute the value loss for this value-head
                    value_loss = tf.reduce_mean(0.5 * tf.square(R - vpred[:-1]))
                else:
                    raise NotImplementedError('Unknown loss type.')

                # entropy loss
                entropy_loss = tf.reduce_mean(head.ent)
                loss_endpoints = {'pg_loss': tf.reduce_mean(pg_loss),
                                  'value_loss': tf.reduce_mean(value_loss),
                                  'entropy_loss': tf.reduce_mean(entropy_loss),
                                  'return': tf.reduce_mean(tf.reduce_mean(vf) if nc.use_loss_type == 'ppo2' else
                                                           inputs.R),
                                  'rms_loss': tf.reduce_mean(rms_loss),
                                  }
                # z gaussian prior loss
                if nc.z_prior_type == 'Gaussian':
                    logpz = log_normal_pdf(z_curr, 0, nc.z_logvar_prior)
                    logqz_x = log_normal_pdf(z_curr, mu, logvar)
                    z_prior_loss = -(logpz - logqz_x)

                    z_prior_loss_weight = nc.z_prior_param1 * (
                        1.0 - (
                        1.0 - tf.math.minimum(1.0, tf.cast(nc.total_timesteps, tf.float32) / nc.z_prior_param2))
                        ** nc.z_prior_param3)
                    z_prior_loss = tf.reduce_mean(z_prior_loss)
                    loss_endpoints.update({
                        'z_prior_loss': tf.reduce_mean(z_prior_loss),
                        'weighted_z_prior_loss': tf.reduce_mean(z_prior_loss) * z_prior_loss_weight,
                        'z_prior_loss_weight': z_prior_loss_weight,
                        'z_curr': tf.reduce_mean(z_curr),
                        'mu': tf.reduce_mean(mu),
                        'logvar': tf.reduce_mean(logvar),
                    })
                    endpoints['z_curr'] = z_curr
                    endpoints['mu'] = mu
                    endpoints['logvar'] = logvar
                elif nc.z_prior_type == 'VQ':
                    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - z_encode) ** 2)
                    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(z_encode)) ** 2)
                    avg_probs = tf.reduce_mean(encodings, 0)
                    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))
                    loss_endpoints.update({
                        'e_latent_loss': tf.reduce_mean(e_latent_loss),
                        'q_latent_loss': tf.reduce_mean(q_latent_loss),
                        'perplexity': tf.reduce_mean(perplexity),
                    })
                else:
                    z_prior_loss = 0
                loss = PMCLosses(
                    total_reg_loss=total_reg_loss,
                    pg_loss=pg_loss,
                    value_loss=value_loss,
                    entropy_loss=entropy_loss,
                    loss_endpoints=loss_endpoints
                )

        # collect vars, endpoints, etc.
        trainable_vars = _make_vars(sc)

    return PMCOutputs(
        self_fed_heads=head,
        outer_fed_heads=head,
        loss=loss,
        vars=trainable_vars,
        endpoints=endpoints,
        value_head=vf,
        ob_rms=ob_rms,
    )


# APIs
net_build_fun = pmc_net
net_config_cls = PMCConfig
net_inputs_placeholders_fun = pmc_inputs_placeholder
