import os
import argparse
import json
import time


import tensorflow as tf
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


def model_fn(features, labels, mode, params):
    pred_stack = gnn.dynamical.dynamical_multisteps(features,
                                                    params,
                                                    params['pred_steps'],
                                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'next_steps': pred_stack}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels, pred_stack)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=100,
            decay_rate=0.99,
            staircase=True,
            name='learning_rate'
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use the loss between adjacent steps in original time_series as baseline
    time_series_loss_baseline = tf.metrics.mean_squared_error(features['time_series'][:, 1:, :, :],
                                                              features['time_series'][:, :-1, :, :])

    eval_metric_ops = {'time_series_loss_baseline': time_series_loss_baseline}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(features, seg_len, pred_steps, batch_size, mode='train'):

    time_series = features['time_series']
    num_sims, time_steps, num_agents, ndims = time_series.shape
    # Shape [num_sims, time_steps, num_agents, ndims]
    time_series_stack = gnn.utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                                    seg_len)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, num_agents, ndims]
    expected_time_series_stack = gnn.utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                             pred_steps)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_agents, ndims]
    assert time_series_stack.shape[:2] == expected_time_series_stack.shape[:2]

    time_segs = time_series_stack.reshape([-1, seg_len, num_agents, ndims])
    expected_time_segs = expected_time_series_stack.reshape([-1, pred_steps, num_agents, ndims])

    processed_features = {'time_series': time_segs}
    if 'edge_type' in features:
        processed_features['edge_type'] = features['edge_type']
    labels = expected_time_segs

    if mode == 'train':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True
        )
    elif mode == 'eval':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            shuffle=False
        )
    elif mode == 'test':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            batch_size=batch_size,
            shuffle=False
        )
