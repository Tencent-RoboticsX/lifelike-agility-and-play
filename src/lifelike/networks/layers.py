import tensorflow as tf


# normalization stuff
def rms(inputs,
        momentum=0.99,
        begin_norm_axis=0,
        moving_mean_initializer=None,
        moving_std_initializer=None,
        trainable=True,
        scope=None):
    """A simple implementation of running mean and std.

    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization.
    In tensorflow's bn, the moving mean and std are updated as
    moving_mean = moving_mean * (1 - momentum) + mean(batch) * momentum
    moving_var = moving_var * (1 - momentum) + var(batch) * momentum
    by writing one as
    moving_mean = moving_mean + momentum * (mean(batch) - moving_mean)
    we can easily find that this is gradient descent of a least square, so we can use
    least square loss to replace this update operation to return a rms_loss, following the TLeague principle

    CAUTION: the first dim is assumed to be the batch dim

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
      begin_norm_axis: beginning dim, default by 0, that is the batch dim
      trainable: if the moving variables are trainable
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, default_name="rms"):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[begin_norm_axis + 1:]

        if moving_mean_initializer:
            moving_mean = tf.get_variable("moving_mean", (1,) + params_shape,
                                          initializer=moving_mean_initializer,
                                          trainable=trainable)
        else:
            moving_mean = tf.get_variable("moving_mean", (1,) + params_shape,
                                          initializer=tf.zeros_initializer(),
                                          trainable=trainable)
        if moving_std_initializer:
            moving_std = tf.get_variable("moving_std", (1,) + params_shape,
                                         initializer=moving_std_initializer,
                                         trainable=trainable)
        else:
            moving_std = tf.get_variable("moving_std", (1,) + params_shape,
                                         initializer=tf.ones_initializer(),
                                         trainable=trainable)

        outputs = (inputs - moving_mean) / (moving_std + 1e-8)
        mean, variance = tf.nn.moments(inputs, axes=list(range(begin_norm_axis + 1)))
        rms_loss = 0.5 * momentum * (tf.square(moving_mean - tf.stop_gradient(mean)) +
                                     tf.square(moving_std - tf.stop_gradient(tf.sqrt(variance))))

    return outputs, rms_loss
