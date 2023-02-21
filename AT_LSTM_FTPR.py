"""
ATST-LSTM reimplemented by Sajal Halder
PhD Candidate, RMIT University, Australia
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import operator
import networkx as nx

from sklearn.preprocessing import MinMaxScaler
import time
import random, collections, itertools
import math
import dataConvert as dc
import ImplementationForSocialData as SD

# Hyparameters



min_users = 2
min_pois = 2

sequence_time_duration = 8*3600 # total 8 hours
queue_prediction = 1

class Config(object):
    """ Config."""
    init_scale = 1
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 1
    num_steps = 0
    hidden_size = 200
    max_epoch = 10
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 0
    regularization = 0.0025
    num_sample = 0
    user_size = 1000
    min_under_demand = 0.3
    envy_cap = 0.3
    alpha = 1.0
    # topk = 5

config = Config()

def data_type():
    return tf.float32


def find_idcg():
    idcg_3, idcg_5, idcg_10 = 0.0, 0.0, 0.0

    for i in range(3):
        idcg_3 = idcg_3 + tf.math.log(2.0) / tf.math.log(float(i) + 2.0)

    for i in range(5):
        idcg_5 = idcg_5 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    for i in range(10):
        idcg_10 = idcg_10 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    return idcg_3, idcg_5, idcg_10


class AT_LSTM(object):

    def __init__(self, config, is_training=True):
        """
        :param is_training:
        """
        self.num_steps = num_steps = config.num_steps
        self.regularization = config.regularization
        size = config.hidden_size
        vocab_size = config.vocab_size

        user_size = config.user_size
        num_sample = config.num_sample

        # print(" vocab_size =", vocab_size, " num_sample = ", num_sample, " user_size = ", user_size)

        self.batch_size = tf.compat.v1.placeholder(tf.int32, [])
        self._input_data = tf.compat.v1.placeholder(tf.int32, [None, num_sample])
        self._input_space = tf.compat.v1.placeholder(tf.float32, [None, num_sample])
        self._input_time = tf.compat.v1.placeholder(tf.float32, [None, num_sample])
        self._user = tf.compat.v1.placeholder(tf.int32, [None, 1])
        self._targets = tf.compat.v1.placeholder(tf.int32, [None,num_sample])
        self._target_time = tf.compat.v1.placeholder(tf.float32, [None, num_sample])
        self._target_space = tf.compat.v1.placeholder(tf.float32, [None, num_sample])

        self.mask_x = tf.compat.v1.placeholder(tf.float32, [None, num_sample])

        self._negative_samples = tf.compat.v1.placeholder(tf.int32, [None, num_sample])
        self._negative_samples_time = tf.compat.v1.placeholder(tf.float32, [None, num_sample])
        self._negative_samples_distance = tf.compat.v1.placeholder(tf.float32, [None, num_sample])
        self.vocabulary_distance = tf.compat.v1.placeholder(tf.float32, [None, config.vocab_size])

        batch_size = self.batch_size

        lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.compat.v1.get_variable("embedding", [config.vocab_size, size], dtype=data_type())
            sequences = tf.compat.v1.nn.embedding_lookup(embedding, self._input_data)
            targets = tf.compat.v1.nn.embedding_lookup(embedding, self._targets)
            negative_samples = tf.compat.v1.nn.embedding_lookup(embedding, self._negative_samples)
            embedding_user = tf.compat.v1.get_variable("embedding_user", [user_size, size], dtype=data_type())
            user = tf.compat.v1.nn.embedding_lookup(embedding_user, self._user)
        state = self._initial_state

        W_p = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_p", [size, size], dtype=data_type()), 0), [batch_size, 1, 1])
        W_t = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_t", [1, size], dtype=data_type()), 0), [batch_size, 1, 1])
        W_s = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_s", [1, size], dtype=data_type()), 0), [batch_size, 1, 1])

        inputs_x = tf.matmul(sequences, W_p) + tf.matmul(tf.expand_dims(self._input_time, 2), W_t) + tf.matmul(tf.expand_dims(self._input_space, 2), W_s)
        target_input = tf.matmul(targets, W_p) + tf.matmul(tf.expand_dims(self._target_time, 2), W_t) + tf.matmul(tf.expand_dims(self._target_space, 2), W_s)
        negative_sample_input = tf.matmul(negative_samples, W_p) + tf.matmul(
            tf.expand_dims(self._negative_samples_time, 2), W_t) + tf.matmul(
            tf.expand_dims(self._negative_samples_distance, 2), W_s)

        if is_training and config.keep_prob < 1:
            inputs_x = tf.compat.v1.nn.dropout(inputs_x, config.keep_prob)

        outputs, states = tf.compat.v1.nn.dynamic_rnn(cell, inputs=inputs_x, initial_state=state, time_major=False)
        self._final_state = states

        W_r = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_r", [size, size], dtype=data_type()), 0), [batch_size, 1, 1])
        W_u = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_u", [size, size], dtype=data_type()), 0), [batch_size, 1, 1])
        queries = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_z", [1, size], dtype=data_type()), 0), [batch_size, 1, 1])
        weights = tf.matmul(queries, tf.transpose(outputs, [0, 2, 1]))  # (batch_size,1, num_steps)
        weights = weights / (size ** 0.5 + 1e-7)
        weights = tf.nn.softmax(weights)  # (batch_size,1, num_steps)
        weights *= self.mask_x[:, None, :]  # broadcasting. (batch_size, 1, num_steps)
        r = tf.matmul(weights, outputs)  # ( batch_size,1, size)

        output_ = tf.matmul(r, W_r) + tf.matmul(user, W_u)  # ( batch_size,1, size)
        output_y = tf.matmul(output_, tf.transpose(target_input, [0, 2, 1]))  # ( batch_size,1, 1)
        output_sample = tf.matmul(output_,tf.transpose(negative_sample_input, [0, 2, 1]))  # ( batch_size,1, num_sample)

        self._lr = tf.Variable(0.0, trainable=False)
        self.tvars = tf.compat.v1.trainable_variables()

        """
        Compute the BPR objective which is sum_uij ln sigma(x_uij) 

        """
        if is_training:
            #ranking_loss = tf.compat.v1.reduce_sum(tf.compat.v1.log(tf.clip_by_value((1.0 + tf.exp(-tf.compat.v1.to_float(tf.tile(output_y, [1, 1, num_sample]) - output_sample))), 1e-8, 1.0) + 1))
            # print("output y = ", output_y.get_shape(), " output sample = ", output_sample.get_shape())
            # ranking_loss = tf.reduce_sum(tf.math.abs(tf.compat.v1.to_float(tf.tile(output_y, [1, 1, num_sample]) - output_sample)))
            ranking_loss = tf.reduce_sum(tf.math.abs(tf.compat.v1.to_float(output_y - output_sample)))

            self.cost = tf.compat.v1.div(ranking_loss, tf.compat.v1.to_float(num_sample * batch_size))

        if not is_training:
            with tf.name_scope("prediction"):

                all_time =  self._target_time #tf.tile(tf.expand_dims(self._target_time, 2), [1, vocab_size, 1])
                # #all_input = tf.matmul(tf.tile(tf.expand_dims(embedding, 0), [batch_size, 1, 1]), W_p) + tf.matmul( all_time, W_t) + tf.matmul(tf.expand_dims(self.vocabulary_distance, 2), W_s)
                # all_input = tf.matmul(all_time, W_t) + tf.matmul(tf.expand_dims(self.vocabulary_distance, 2), W_s)
                #
                # logits = tf.reshape(tf.matmul(output_, tf.transpose(all_input, [0, 2, 1])),[batch_size, -1])  # ( batch_size,vocab_size)

                # all_time = tf.expand_dims(self._target_time, 2) #, [1, vocab_size, 1])
                # print("all time shape ", all_time.get_shape(), " distance ", self.vocabulary_distance.get_shape())
                a1= tf.matmul(tf.tile(tf.expand_dims(embedding, 0), [batch_size, 1, 1]), W_p)
                # b1 = tf.matmul(all_time, W_t)
                c1 = tf.matmul(tf.expand_dims(self.vocabulary_distance, 2), W_s)
                # print("a1 shape ", a1.get_shape(), " b1  ", b1.get_shape(), " c1 = ", c1.get_shape())
                all_input = a1 +  c1
                logits = tf.reshape(tf.matmul(output_, tf.transpose(all_input, [0, 2, 1])),
                                    [batch_size, -1])  # ( batch_size,vocab_size)

                self._list = tf.nn.top_k(logits,min(vocab_size,25))[1]
                self.list_value = logits

                self.prediction_5 = tf.nn.top_k(logits, 5)[1]
                self.prediction_10 = tf.nn.top_k(logits, 10)[1]

                # self.acc_1 = tf.reduce_sum(tf.cast(tf.equal(tf.tile(self._targets, [1, 1]), tf.nn.top_k(logits, 1)[1]), tf.float32)) / tf.cast(batch_size, tf.float32)
                # self.acc_3 = tf.reduce_sum(tf.cast(tf.equal(tf.tile(self._targets, [1, 3]), tf.nn.top_k(logits, 3)[1]), tf.float32)) / tf.cast(batch_size, tf.float32)

                # print(self._targets)
                self.acc_5 = tf.reduce_sum(tf.cast(tf.equal(self._targets[:,0:5], tf.nn.top_k(logits, 5)[1]), tf.float32)) / tf.cast(batch_size, tf.float32)
                self.acc_10 = tf.reduce_sum(tf.cast(tf.equal(self._targets[:,0:10], tf.nn.top_k(logits, 10)[1]),tf.float32)) / tf.cast(batch_size, tf.float32)

                # self.acc_5 = tf.reduce_sum(tf.cast(tf.equal(tf.tile(self._targets [1, 5]), tf.nn.top_k(logits, 5)[1]), tf.float32)) / tf.cast(batch_size, tf.float32)
                # self.acc_10 = tf.reduce_sum(tf.cast(tf.equal(tf.tile(self._targets, [1, 10]), tf.nn.top_k(logits, 10)[1]), tf.float32)) / tf.cast(batch_size, tf.float32)


                idcg_3, idcg_5, idcg_10 = find_idcg()

                # self.ndcg_3 = tf.reduce_sum((tf.math.log(2.0) / (tf.math.log(
                #     tf.cast(tf.where(tf.cast(tf.equal(self._targets, tf.nn.top_k(logits, 3)[1]), tf.int64)),
                #             tf.float32) + 2.0))) / idcg_3) / tf.cast(batch_size, tf.float32)
                self.ndcg_5 = tf.reduce_sum((1 / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(self._targets[:,0:5], tf.nn.top_k(logits, 5)[1]), tf.int64)),tf.float32) + 2.0) / tf.math.log(2.0))) / idcg_5) / tf.cast(batch_size, tf.float32)
                self.ndcg_10 = tf.reduce_sum((1 / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(self._targets[:,0:10], tf.nn.top_k(logits, 10)[1]), tf.int64)), tf.float32) + 2.0) / tf.math.log(2.0))) / idcg_10) / tf.cast(batch_size, tf.float32)

                expand_targets = self._targets[:,0:5]
                isequal = tf.equal(expand_targets, self.prediction_5)

                # #print(np.asfarray(expand_targets)[:3])
                #
                correct_prediction_5 = tf.reduce_sum(tf.cast(isequal, tf.float32))

                self.precison_5 = correct_prediction_5 / tf.cast(batch_size * 5, tf.float32)
                self.recall_5 = correct_prediction_5 / tf.cast(batch_size, tf.float32)         # Need discussion about the code
                self.f1_5 = 2 * self.precison_5 * self.recall_5 / (self.precison_5 + self.recall_5 + 1e-10)

                expand_targets = self._targets[:,0:10]
                isequal = tf.equal(expand_targets, self.prediction_10)

                correct_prediction_10 = tf.reduce_sum(tf.cast(isequal, tf.float32))
                self.precison_10 = correct_prediction_10 / tf.cast(batch_size * 10, tf.float32)

                self.recall_10 = correct_prediction_10 / tf.cast(batch_size, tf.float32)   # Need discussion about this code
                self.f1_10 = 2 * self.precison_10 * self.recall_10 / (self.precison_10 + self.recall_10 + 1e-10)

        if not is_training:
            return

        # optiminzer
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), config.max_grad_norm)
        optimizer = tf.compat.v1.train.AdagradOptimizer(self._lr)
        self.train_op = optimizer.apply_gradients(zip(grads, self.tvars))

        self._new_lr = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.compat.v1.assign(self._lr, self._new_lr)

    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_new_num_steps(self, session, num_steps_value):
        session.run(self._num_step_update, feed_dict={self.new_num_steps: num_steps_value})

    def initial_state(self):
        return self._initial_state

    def final_state(self):
        return self._final_state


def evaluate(model, session, data, vocabulary_distances, batch_size):
    total_num = 0


    total_acc_5 = 0.0
    total_acc_10 = 0.0

    total_NDCG_5 = 0.0
    total_NDCG_10 = 0.0

    total_list = []
    total_value = []

    total_precison_5, total_precison_10 = 0.0, 0.0
    total_f1_5, total_f1_10 = 0.0, 0.0

    state = session.run(model._initial_state, feed_dict={model.batch_size: batch_size})
    fetches = [model.list_value, model._list, model.precison_5, model.precison_10, model.f1_5, model.f1_10, model.acc_5, model.acc_10, model.ndcg_5, model.ndcg_10,model._final_state]
    # fetches = [model._list, model.final_state]

    for step, (return_sequence_x, return_sequence_y, return_time_x, return_time_y, return_distance_x,
               return_distance_y, return_mask_x, return_vocabulary_distances, return_user) in enumerate(
        batch_iter(data, vocabulary_distances, batch_size)):
        total_num = total_num + 1

        feed_dict = {}
        feed_dict[model.batch_size] = batch_size
        feed_dict[model._input_data] = return_sequence_x
        feed_dict[model._input_space] = return_distance_x.astype(np.float32)
        feed_dict[model._input_time] = return_time_x.astype(np.float32)
        return_user = np.array(return_user)
        return_user = np.reshape(return_user, [return_user.shape[0], 1])
        feed_dict[model._user] = return_user
        feed_dict[model._targets] = return_sequence_y
        feed_dict[model._target_time] = return_time_y
        feed_dict[model._target_space] = return_distance_y
        feed_dict[model.mask_x] = return_mask_x
        feed_dict[model.vocabulary_distance] = [x.astype(np.float32) for x in return_vocabulary_distances]

        for i, (c, h) in enumerate(model._initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        list_value, list, c_pre_5, c_pre_10, c_f1_5, c_f1_10,  c_acc_5, c_acc_10, c_ndcg_5, c_ndcg_10, state = session.run(fetches, feed_dict)

        # print( "list = ", list)

        total_list += [l for l in list]
        total_value += [l for l in list_value]

        total_acc_5 = total_acc_5 + c_acc_5
        total_acc_10 = total_acc_10 + c_acc_10


        total_NDCG_5 = total_NDCG_5 + c_ndcg_5
        total_NDCG_10 = total_NDCG_10 + c_ndcg_10

        total_precison_5 = total_precison_5 + c_pre_5
        total_precison_10 = total_precison_10 + c_pre_10

        total_f1_5 = total_f1_5 + c_f1_5
        total_f1_10 = total_f1_10 + c_f1_10


    total_acc_5 = total_acc_5 / total_num
    total_acc_10 = total_acc_10 / total_num

    total_NDCG_5 = total_NDCG_5 / total_num
    total_NDCG_10 = total_NDCG_10 / total_num

    total_precison_5 = total_precison_5 / total_num
    total_precison_10 = total_precison_10/ total_num

    total_f1_5 = total_f1_5 / total_num
    total_f1_10 = total_f1_10 / total_num



    return total_value, total_list, total_precison_5, total_precison_10, total_f1_5, total_f1_10, total_acc_5, total_acc_10, total_NDCG_5, total_NDCG_10
    # return list

def run_epoch(model, session, data, negative_samples, global_steps, batch_size):
    """Runs the model on the given data."""

    state = session.run(model._initial_state, feed_dict={model.batch_size: batch_size})
    fetches = [model.cost, model._final_state, model.train_op]

    for step, (return_sequence_x, return_sequence_y, return_time_x, return_time_y, return_distance_x,
               return_distance_y, return_mask_x,
               return_negative_sample, return_negative_time_sample, return_negative_distance_sample,
               return_user) in enumerate(batch_iter_sample(data, negative_samples, batch_size)):

        feed_dict = {}
        feed_dict[model.batch_size] = batch_size

        # print(" _input _data", return_sequence_x)
        feed_dict[model._input_data] = return_sequence_x
        feed_dict[model._input_space] = return_distance_x.astype(np.float32)
        feed_dict[model._input_time] = return_time_x.astype(np.float32)
        return_user = np.array(return_user)
        feed_dict[model._user] = np.reshape(return_user, [return_user.shape[0], 1])
        feed_dict[model._targets] = return_sequence_y
        feed_dict[model._target_time] = return_time_y
        feed_dict[model._target_space] = return_distance_y
        feed_dict[model.mask_x] = return_mask_x
        feed_dict[model._negative_samples] = return_negative_sample
        feed_dict[model._negative_samples_time] = return_negative_time_sample.astype(np.float32)
        feed_dict[model._negative_samples_distance] = return_negative_distance_sample.astype(np.float32)

        for i, (c, h) in enumerate(model._initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        cost, state, _ = session.run(fetches, feed_dict)
        # print("Cost = ",cost)

        # if (global_steps % 100== 0):
        #     print("the %i step, train cost is: %f" % (global_steps, cost))
        global_steps += 1

    return global_steps, cost


def euclidean_dist(point1, point2):
    (lat1, lon1) = point1
    (lat2, lon2) = point2
    radius = 6371000  # m

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def _build_sequence_SD(userlocation, coordinates):
    user_voc = collections.Counter(userlocation[0, :].tolist())
    # print(user_voc.keys())
    sequence = []
    sequence_user = []
    sequence_time = []
    sequence_distance = []
    sequence_queue = []

    sum_sequence = 0
    start_time_of_seq = 0
    for user in user_voc.keys():

        checkin_user_radex = np.argwhere(userlocation[0, :] == user)
        # print(checkin_user_radex)
        checkin_user_all = userlocation[:, checkin_user_radex[:, 0]]
        # print(checkin_user_all)

        user_count = 0
        sequence_location = []
        sequence_time_user = []
        sequence_distance_user = []
        sequence_queue_time = []

        temporal_sequence_location = []
        temporal_sequence_time_user = []
        temporal_sequence_distance_user = []
        temporal_sequence_queue_time = []

        sorted_time = np.sort(checkin_user_all[1, :])
        sorted_time_index = np.argsort(checkin_user_all[1, :])  # Find index based on sorted time

        pre_poi_index = 0
        check_out_time = 0
        for i in range(len(checkin_user_radex)):
            if i == 0:
                sequence_location.append(checkin_user_all[2, sorted_time_index[i]])  # Add POI based on time
                sequence_time_user.append(100)
                sequence_distance_user.append(1)
                sequence_queue_time.appen(100)
                start_time_of_seq = sorted_time[i]

            else:
                if sorted_time[i] - start_time_of_seq > sequence_time_duration: # sorted_time[pre_poi_index] > 28800:  # 21600:   # Sequence length 8 hours
                    if len(set(sequence_location)) >= min_pois:  # Number of poi is >=5
                        sequence_location = list(map(int, sequence_location))
                        sequence_time_user = list(map(int, sequence_time_user))
                        sequence_queue_time = list(map(int, sequence_queue_time))

                        temporal_sequence_location.append(sequence_location)
                        temporal_sequence_time_user.append(sequence_time_user)
                        temporal_sequence_distance_user.append(sequence_distance_user)
                        temporal_sequence_queue_time.append(sequence_queue_time)

                        user_count = user_count + 1
                    sequence_location = []
                    sequence_time_user = []
                    sequence_distance_user = []
                    sequence_queue_time = []

                    sequence_location.append(checkin_user_all[2, sorted_time_index[i]])  # Add POI
                    sequence_time_user.append(100)
                    sequence_distance_user.append(1)
                    sequence_queue_time.appen(100)
                    pre_poi_index = i
                    start_time_of_seq = sorted_time[i]
                else:
                    if checkin_user_all[2, sorted_time_index[i]] not in sequence_location:
                        sequence_location.append(checkin_user_all[2, sorted_time_index[i]])
                        sequence_time_user.append(sorted_time[i] - sorted_time[pre_poi_index] + 1)
                        distance = euclidean_dist(coordinates[checkin_user_all[2, sorted_time_index[i]]],
                                                  coordinates[checkin_user_all[2, sorted_time_index[pre_poi_index]]])
                        sequence_distance_user.append(distance + 1e-5)
                        sequence_queue_time.append(sorted_time[i] - sorted_time[check_out_time] + 1)
                        pre_poi_index = i
                    else:
                        check_out_time = sorted_time[i]

        sum_sequence = sum_sequence + 1

        if user_count >= min_users:  # prune based on min_user
            sequence = sequence + temporal_sequence_location
            sequence_time = sequence_time + temporal_sequence_time_user
            sequence_distance = sequence_distance + temporal_sequence_distance_user
            sequence_user = sequence_user + [user] * user_count
            sequence_queue = sequence_queue + temporal_sequence_queue_time

    max_time = max([max(x) for x in sequence_time])
    max_distance = max([max(x) for x in sequence_distance])
    sequence_time = [[y / max_time for y in x] for x in sequence_time]  # Normalize (0,1)
    sequence_distance = [[y / max_distance for y in x] for x in sequence_distance]  # Normaliaze (0,1)

    max_queue = max([max(x) for x in sequence_queue])
    sequence_queue = [[y / max_queue for y in x] for x in sequence_queue]  # Normalize (0,1)

    max_len = max([len(x) for x in sequence])

    return sequence, sequence_user, sequence_time, sequence_distance, max_len, sequence_queue


def _build_sequence(userlocation, dfNodes):
    user_voc = collections.Counter(userlocation[0, :].tolist())
    # print(user_voc.keys())
    sequence = []
    sequence_user = []



    sequence_time = []
    sequence_distance = []
    sequence_queue = []

    sum_sequence = 0
    start_time_of_seq = 0
    check_out_time = 0
    for user in user_voc.keys():

        checkin_user_radex = np.argwhere(userlocation[0, :] == user)
        # print(checkin_user_radex)
        checkin_user_all = userlocation[:, checkin_user_radex[:, 0]]
        # print(checkin_user_all)

        user_count = 0
        sequence_location = []
        sequence_time_user = []
        sequence_distance_user = []
        sequence_queue_time = []

        temporal_sequence_location = []
        temporal_sequence_time_user = []
        temporal_sequence_distance_user = []
        temporal_sequence_queue_time = []

        sorted_time = np.sort(checkin_user_all[1, :])
        sorted_time_index = np.argsort(checkin_user_all[1, :])  # Find index based on sorted time

        pre_poi_index = 0
        check_out_time = 0
        for i in range(len(checkin_user_radex)):
            if i == 0:
                sequence_location.append(checkin_user_all[2, sorted_time_index[i]])  # Add POI based on time
                sequence_time_user.append(100)
                sequence_distance_user.append(1)
                start_time_of_seq = sorted_time[i]
                sequence_queue_time.append(100)

            else:
                if sorted_time[i] -start_time_of_seq > sequence_time_duration: # = sorted_time[i] sorted_time[pre_poi_index] > 28800:  # 21600:   # Sequence length 8 hours
                    if len(set(sequence_location)) >= min_pois:  # Number of poi is >=5
                        sequence_location = list(map(int, sequence_location))
                        sequence_time_user = list(map(int, sequence_time_user))
                        sequence_queue_time = list(map(int, sequence_queue_time))

                        temporal_sequence_location.append(sequence_location)
                        temporal_sequence_time_user.append(sequence_time_user)
                        temporal_sequence_distance_user.append(sequence_distance_user)
                        temporal_sequence_queue_time.append(sequence_queue_time)

                        user_count = user_count + 1
                    sequence_location = []
                    sequence_time_user = []
                    sequence_distance_user = []
                    sequence_queue_time = []

                    sequence_location.append(checkin_user_all[2, sorted_time_index[i]])  # Add POI
                    sequence_time_user.append(100)
                    sequence_distance_user.append(1)
                    sequence_queue_time.append(100)
                    pre_poi_index = i
                    start_time_of_seq = sorted_time[i]
                else:
                    if checkin_user_all[2, sorted_time_index[i]] not in sequence_location:
                        sequence_location.append(checkin_user_all[2, sorted_time_index[i]])
                        sequence_time_user.append(sorted_time[i] - sorted_time[pre_poi_index] + 1)
                        distance = int(dfNodes[(dfNodes.to == checkin_user_all[2, sorted_time_index[i]]) & (
                                dfNodes.from1 == checkin_user_all[2, sorted_time_index[pre_poi_index]])]['distance'])
                        sequence_distance_user.append(distance + 1e-5)
                        sequence_queue_time.append(sorted_time[i] - sorted_time[check_out_time] + 1)
                        pre_poi_index = i
                        check_out_time = i # if check out is not available
                    else:
                        check_out_time = i

        sum_sequence = sum_sequence + 1

        if user_count >= 1:  # prune based on min_user
            sequence = sequence + temporal_sequence_location
            sequence_time = sequence_time + temporal_sequence_time_user
            sequence_distance = sequence_distance + temporal_sequence_distance_user
            sequence_user = sequence_user + [user] * user_count
            sequence_queue = sequence_queue + temporal_sequence_queue_time

    max_time = max([max(x) for x in sequence_time])
    max_distance = max([max(x) for x in sequence_distance])
    sequence_time = [[y / max_time for y in x] for x in sequence_time]  # Normalize (0,1)
    sequence_distance = [[y / max_distance for y in x] for x in sequence_distance]  # Normaliaze (0,1)

    max_len = max([len(x) for x in sequence])

    max_queue = max([max(x) for x in sequence_queue])
    sequence_queue = [[y / max_queue for y in x] for x in sequence_queue]  # Normalize (0,1)

    return sequence, sequence_user, sequence_time, sequence_distance, max_len, sequence_queue


def new_build_location_voc(sequence, dfNodes):
    locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
    # print(locations_voc)
    locations_list = list(locations_voc.keys())
    # print(location_list)

    newsequence = []
    word_to_id = dict(zip(locations_list, range(len(locations_list))))
    # print(word_to_id)

    for lst in sequence:
        newsequence.append([word_to_id[x] for x in lst])

    citys = set(dfNodes.from1)  # locations[:,0].tolist())
    print(citys)

    clusters = []

    city_voc = collections.Counter(citys)
    # print(city_voc)
    city_list = list(city_voc.keys())
    # print("city list ", city_list)
    city_to_id = dict(zip(city_list, range(len(city_list))))
    # print(city_to_id)
    citys_id = [city_to_id[word] for word in citys]

    for i in range(len(city_list)):
        clusters.append([n for n in range(len(citys_id)) if citys_id[n] == i])

    return newsequence, clusters


def new_build_location_voc_SD(sequence, coordinates):
    locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
    # print(locations_voc)
    locations_list = list(locations_voc.keys())
    # print(location_list)

    newsequence = []
    word_to_id = dict(zip(locations_list, range(len(locations_list))))
    # print(word_to_id)

    for lst in sequence:
        newsequence.append([word_to_id[x] for x in lst])

    citys = list(coordinates.keys())  # set(locations[:,0].tolist())
    print(citys)

    clusters = []

    city_voc = collections.Counter(citys)
    # print(city_voc)
    city_list = list(city_voc.keys())
    # print("city list ", city_list)
    city_to_id = dict(zip(city_list, range(len(city_list))))
    # print(city_to_id)
    citys_id = [city_to_id[word] for word in citys]

    for i in range(len(city_list)):
        clusters.append([n for n in range(len(citys_id)) if citys_id[n] == i])

    return newsequence, clusters


def pop_n(sequence, k):
    locations_voc = collections.Counter(list(itertools.chain.from_iterable(sequence)))
    # print(locations_voc)
    sorted_locations_voc = sorted(locations_voc.items(), key=lambda d: d[1], reverse=True)
    # print(sorted_locations_voc)
    item = [counter for counter, value in enumerate(sorted_locations_voc) if counter < k]
    # print("Item = ",item)
    return item


def padding(x, y, new_x, new_y, max_len):
    for i, (x, y) in enumerate(zip(x, y)):
        if len(x) <= max_len:
            new_x[i, 0:len(x)] = x
            new_y[i] = y
        else:
            new_x[i] = (x[0:max_len])
            new_y[i] = y

    new_set = (new_x, new_y)
    del new_x, new_y
    return new_set


def padding3(new_y, x, max_len):
    for i, value in enumerate(x):
        if len(value) < max_len:
            new_y[i,0:len(value)] = value
        else:
            new_y[i] = (value[0:max_len])

    return new_y



def padding1(x, y, new_x, new_y, max_len):
    for i, (x, y) in enumerate(zip(x, y)):
        if len(x) <= max_len:
            new_x[i, 0:len(x)] = x
        else:
            new_x[i] = (x[0:max_len])


        if len(y) <= max_len:
            new_y[i, 0:len(y)] = y

        else:
            new_y[i] = (y[0:max_len])


    new_set = (new_x, new_y)
    del new_x, new_y
    return new_set


def generate_mask(x, max_len):
    new_mask_x = np.zeros([len(x), max_len])
    for i, y in enumerate(x):
        if len(y) <= max_len:
            new_mask_x[i, 0:len(y)] = 1
        else:
            new_mask_x[i, :] = 1

    return new_mask_x


def generate_negative_sample(l, num_sample, clusters, top_500):
    cluster_j = l
    n_samples = len(clusters[cluster_j])

    if n_samples >= num_sample:
        index = random.sample(range(n_samples), num_sample)
        lastindex = [clusters[cluster_j][i] for i in index]

        if l in lastindex:
            index = random.sample(range(n_samples), num_sample)
            lastindex = [clusters[cluster_j][i] for i in index]
    else:
        if n_samples > l:
            lastindex = list(set(clusters[cluster_j]) ^ set([int(l)])) + top_500[0:num_sample - n_samples + 1]
        else:
            lastindex = list(set(clusters[cluster_j]) ^ set([int(l)])) + top_500[0:num_sample - n_samples + 1]


    return lastindex


def generate_negative_sample_SD(l, num_sample, clusters, top_500):
    cluster_j = l
    n_samples = len(clusters[cluster_j])

    if n_samples >= num_sample:
        index = random.sample(range(n_samples), num_sample)
        lastindex = [clusters[cluster_j][i] for i in index]

        if l in lastindex:
            index = random.sample(range(n_samples), num_sample)
            lastindex = [clusters[cluster_j][i] for i in index]
    else:
        if n_samples > l:
            lastindex = list(set(clusters[cluster_j]) ^ set([int(l)])) + top_500[0:num_sample - n_samples + 1]
        else:

            lastindex = list(set(clusters[cluster_j]) ^ set([int(l)])) + top_500[0:num_sample - n_samples + 1]


    return lastindex


def padding_negative_sample(targets, negative_sample, negative_distance_sample, clusters, sequence, dfNodes):
    sequence_num, num_sample1 = negative_sample.shape

    for i in range(sequence_num):
        sample = generate_negative_sample(targets[i][-1], num_sample1, clusters, sequence)
        negative_sample[i, :] = np.mat(sample[0:num_sample1])

    for i in range(sequence_num):

        target_location = targets[i][-1] + 1
        for j in range(num_sample1):
            c_location = int(negative_sample[i][j]) + 1
            if (target_location != c_location):

                if len(dfNodes[(dfNodes.from1 == c_location) & (dfNodes.to == target_location)]) > 0:
                    distance = dfNodes[(dfNodes.from1 == c_location) & (dfNodes.to == target_location)].distance
                else:
                    distance = 0.0
                negative_distance_sample[i, j] = distance  # haversine(target_location, c_location)

    return negative_sample, negative_distance_sample


def padding_negative_sample_SD(targets, negative_sample, negative_distance_sample, clusters, sequence, coordinates):
    sequence_num, num_sample1 = negative_sample.shape

    for i in range(sequence_num):
        sample = generate_negative_sample_SD(targets[i][-1], num_sample1, clusters, sequence)
        negative_sample[i, :] = np.mat(sample[0:num_sample1])

    for i in range(sequence_num):

        target_location = targets[i][-1] + 1
        for j in range(num_sample1):
            c_location = int(negative_sample[i][j]) + 1
            if (target_location != c_location):
                distance = euclidean_dist(coordinates[c_location], coordinates[target_location])
                negative_distance_sample[i, j] = distance  # haversine(target_location, c_location)

    return negative_sample, negative_distance_sample


def padding_vocabulary_distance(targets, n_classes, dfNodes):
    vocabulary_distance = np.zeros([len(targets), n_classes + 1])
    sequence_num, voc_size = vocabulary_distance.shape
    for i in range(sequence_num):

        target_location = targets[i][-1] + 1

        for j in range(len(targets[i])):
            c_location = int(targets[i][j]) + 1

            if (target_location != c_location):
                if len(dfNodes[(dfNodes.from1 == c_location) & (dfNodes.to == target_location)]) > 0:
                    vocabulary_distance[i, j] = dfNodes[(dfNodes.from1 == c_location) & (
                                dfNodes.to == target_location)].distance  # haversine(target_location, c_location)
                else:
                    vocabulary_distance[i, j] = 0.0
    return vocabulary_distance


def load_data(train_set, num_sample, clusters, top_13, n_classes, dfNodes, test_portion, max_len, sort_by_len=True):
    (train_set_sequence, sequence_user, train_set_time, train_set_distance) = train_set
    # max_len = max([len(x) for x in train_set_sequence])

    new_sequence = []
    new_sequence_user = []
    new_time = []
    new_distance = []

    # Data augmentation
    for k in range(len(train_set_sequence)):
        new_sequence.append(train_set_sequence[k])
        new_sequence_user.append(sequence_user[k])
        new_time.append(train_set_time[k])
        new_distance.append(train_set_distance[k])
        # for i in range(len(train_set_sequence[k]) - 2):
        #     new_sequence.append(train_set_sequence[k][max(0,i-25):i + 3])
        #     new_sequence_user.append(sequence_user[k])
        #     new_time.append(train_set_time[k][max(0,i-25):i + 3])
        #     new_distance.append(train_set_distance[k][max(0,i-25):i + 3])

    print(" Generate the train set and test set")
    n_samples = len(new_sequence)
    sidx = np.random.permutation(n_samples)
    # print(sidx)

    n_train = int(np.round(n_samples * (1. - test_portion)))
    # print(n_train)

    test_set_sequence = [new_sequence[s] for s in sidx[n_train:]]
    test_set_time = [new_time[s] for s in sidx[n_train:]]
    test_set_distance = [new_distance[s] for s in sidx[n_train:]]
    test_set_user = [new_sequence_user[s] for s in sidx[n_train:]]

    train_set_sequence = [new_sequence[s] for s in sidx[:n_train]]
    train_set_time = [new_time[s] for s in sidx[: n_train]]
    train_set_distance = [new_distance[s] for s in sidx[:n_train]]
    train_set_user = [new_sequence_user[s] for s in sidx[:n_train]]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_sequence)
        test_set_sequence = [test_set_sequence[i] for i in sorted_index]
        test_set_time = [test_set_time[i] for i in sorted_index]
        test_set_distance = [test_set_distance[i] for i in sorted_index]
        test_set_user = [test_set_user[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_sequence)
        train_set_sequence = [train_set_sequence[i] for i in sorted_index]
        train_set_time = [train_set_time[i] for i in sorted_index]
        train_set_distance = [train_set_distance[i] for i in sorted_index]
        train_set_user = [train_set_user[i] for i in sorted_index]



    test_set_sequence_x = [x[0:int(len(x)*0.7)] for x in test_set_sequence]
    test_set_time_x = [x[0:int(len(x)*0.7)] for x in test_set_time]
    test_set_distance_x = [x[0:int(len(x)*0.7)] for x in test_set_distance]
    train_set_sequence_x = [x[0:int(len(x)*0.7)] for x in train_set_sequence]
    train_set_time_x = [x[0:int(len(x)*0.7)] for x in train_set_time]
    train_set_distance_x = [x[0:int(len(x)*0.7)] for x in train_set_distance]

    test_set_sequence_y = [x[int(len(x)*0.7):len(x)] for x in test_set_sequence]
    test_set_time_y = [x[int(len(x)*0.7):len(x)] for x in test_set_time]
    test_set_distance_y = [x[int(len(x)*0.7):len(x)] for x in test_set_distance]
    train_set_sequence_y = [x[int(len(x)*0.7):len(x)] for x in train_set_sequence]
    train_set_time_y = [x[int(len(x)*0.7):len(x)] for x in train_set_time]
    train_set_distance_y = [x[int(len(x)*0.7):len(x)] for x in train_set_distance]

    new_test_set_sequence_x = np.zeros([len(test_set_sequence_x), max_len])
    new_test_set_time_x = np.zeros([len(test_set_time_x), max_len])
    new_test_set_distance_x = np.zeros([len(test_set_distance_x), max_len])
    new_train_set_sequence_x = np.zeros([len(train_set_sequence_x), max_len])
    new_train_set_time_x = np.zeros([len(train_set_time_x), max_len])
    new_train_set_distance_x = np.zeros([len(train_set_distance_x), max_len])

    new_test_set_sequence_y = np.zeros([len(test_set_sequence_y), max_len])
    new_test_set_time_y = np.zeros([len(test_set_time_y), max_len])
    new_test_set_distance_y = np.zeros([len(test_set_distance_y), max_len])
    new_train_set_sequence_y = np.zeros([len(train_set_sequence_y), max_len])
    new_train_set_time_y = np.zeros([len(train_set_time_y), max_len])
    new_train_set_distance_y = np.zeros([len(train_set_distance_y), max_len])

    negative_sample = np.zeros([len(new_train_set_sequence_y), num_sample])
    negative_time_sample = np.zeros([len(new_train_set_sequence_y), num_sample])
    negative_distance_sample = np.zeros([len(new_train_set_sequence_y), num_sample])

    print("Begin the padding process")
    new_train_set_sequence = padding1(train_set_sequence_x, train_set_sequence_y, new_train_set_sequence_x,
                                     new_train_set_sequence_y, max_len)
    new_train_set_time = padding1(train_set_time_x, train_set_time_y, new_train_set_time_x, new_train_set_time_y,
                                 max_len)
    new_train_set_distance = padding1(train_set_distance_x, train_set_distance_y, new_train_set_distance_x,
                                     new_train_set_distance_y, max_len)

    mask_train_x = generate_mask(train_set_sequence_x, max_len)

    new_test_set_sequence = padding1(test_set_sequence_x, test_set_sequence_y, new_test_set_sequence_x,
                                    new_test_set_sequence_y, max_len)

    new_test_set_time = padding1(test_set_time_x, test_set_time_y, new_test_set_time_x, new_test_set_time_y, max_len)

    new_test_set_distance = padding1(test_set_distance_x, test_set_distance_y, new_test_set_distance_x,
                                    new_test_set_distance_y, max_len)

    mask_test_x = generate_mask(test_set_sequence_x, max_len)

    negative_samples, negative_distance_samples = padding_negative_sample(train_set_sequence_x, negative_sample,
                                                                          negative_distance_sample, clusters, top_13,
                                                                          dfNodes)
    #
    # print("negative_time ", negative_time_sample)
    # # print("train time = ", train_set_time)
    # for i in range(num_sample):
    #     negative_time_sample[:, i] = train_set_time_y

    negative_time_sample = padding3(negative_time_sample, train_set_time_y, max_len)

    vocabulary_distances = padding_vocabulary_distance(test_set_sequence_x, n_classes, dfNodes)
    final_train_set = (new_train_set_sequence, new_train_set_time, new_train_set_distance, mask_train_x, train_set_user)
    final_test_set = (new_test_set_sequence, new_test_set_time, new_test_set_distance, mask_test_x, test_set_user)
    final_negative_samples = (negative_samples, negative_time_sample, negative_distance_samples)
    return final_train_set, final_test_set, final_negative_samples, vocabulary_distances



def batch_iter(data, vocabulary_distances, batch_size):
    sequence, time, distance, mask_x, user = data
    sequence_x, sequence_y = sequence
    time_x, time_y = time
    distance_x, distance_y = distance
    data_size = len(sequence_x)

    num_batches_per_epoch = int(data_size / batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_sequence_x = sequence_x[start_index:end_index, :]
        return_sequence_y = sequence_y[start_index:end_index, :]

        return_time_x = time_x[start_index:end_index, :]
        return_time_y = time_y[start_index:end_index, :]

        return_distance_x = distance_x[start_index:end_index, :]
        return_distance_y = distance_y[start_index:end_index, :]

        return_vocabulary_distances = vocabulary_distances[start_index:end_index, :]
        return_mask_x = mask_x[start_index:end_index, :]
        return_user = user[start_index:end_index]

        yield (return_sequence_x, return_sequence_y, return_time_x, return_time_y, return_distance_x,
               return_distance_y, return_mask_x, return_vocabulary_distances, return_user)


def load_data_SD(train_set, coordinates, num_sample, clusters, top_13, n_classes, test_portion, sort_by_len):
    (train_set_sequence, sequence_user, train_set_time, train_set_distance) = train_set
    max_len = max([len(x) for x in train_set_sequence])

    new_sequence = []
    new_sequence_user = []
    new_time = []
    new_distance = []

    # Data augmentation
    for k in range(len(train_set_sequence)):
        for i in range(len(train_set_sequence[k]) - 2):
            new_sequence.append(train_set_sequence[k][0:i + 3])
            new_sequence_user.append(sequence_user[k])
            new_time.append(train_set_time[k][0:i + 3])
            new_distance.append(train_set_distance[k][0:i + 3])

    print(" Generate the train set and test set")
    n_samples = len(new_sequence)
    sidx = np.random.permutation(n_samples)
    # print(sidx)

    n_train = int(np.round(n_samples * (1. - test_portion)))
    # print(n_train)

    test_set_sequence = [new_sequence[s] for s in sidx[n_train:]]
    test_set_time = [new_time[s] for s in sidx[n_train:]]
    test_set_distance = [new_distance[s] for s in sidx[n_train:]]
    test_set_user = [new_sequence_user[s] for s in sidx[n_train:]]

    train_set_sequence = [new_sequence[s] for s in sidx[:n_train]]
    train_set_time = [new_time[s] for s in sidx[: n_train]]
    train_set_distance = [new_distance[s] for s in sidx[:n_train]]
    train_set_user = [new_sequence_user[s] for s in sidx[:n_train]]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_sequence)
        test_set_sequence = [test_set_sequence[i] for i in sorted_index]
        test_set_time = [test_set_time[i] for i in sorted_index]
        test_set_distance = [test_set_distance[i] for i in sorted_index]
        test_set_user = [test_set_user[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_sequence)
        train_set_sequence = [train_set_sequence[i] for i in sorted_index]
        train_set_time = [train_set_time[i] for i in sorted_index]
        train_set_distance = [train_set_distance[i] for i in sorted_index]
        train_set_user = [train_set_user[i] for i in sorted_index]

    test_set_sequence_x = [x[0:len(x) - 1] for x in test_set_sequence]
    test_set_time_x = [x[0:len(x) - 1] for x in test_set_time]
    test_set_distance_x = [x[0:len(x) - 1] for x in test_set_distance]
    train_set_sequence_x = [x[0:len(x) - 1] for x in train_set_sequence]
    train_set_time_x = [x[0:len(x) - 1] for x in train_set_time]
    train_set_distance_x = [x[0:len(x) - 1] for x in train_set_distance]

    test_set_sequence_y = [x[len(x) - 1] for x in test_set_sequence]
    test_set_time_y = [x[len(x) - 1] for x in test_set_time]
    test_set_distance_y = [x[len(x) - 1] for x in test_set_distance]
    train_set_sequence_y = [x[len(x) - 1] for x in train_set_sequence]
    train_set_time_y = [x[len(x) - 1] for x in train_set_time]
    train_set_distance_y = [x[len(x) - 1] for x in train_set_distance]

    new_test_set_sequence_x = np.zeros([len(test_set_sequence_x), max_len])
    new_test_set_time_x = np.zeros([len(test_set_time_x), max_len])
    new_test_set_distance_x = np.zeros([len(test_set_distance_x), max_len])
    new_train_set_sequence_x = np.zeros([len(train_set_sequence_x), max_len])
    new_train_set_time_x = np.zeros([len(train_set_time_x), max_len])
    new_train_set_distance_x = np.zeros([len(train_set_distance_x), max_len])

    new_test_set_sequence_y = np.zeros([len(test_set_sequence_y), 1])
    new_test_set_time_y = np.zeros([len(test_set_time_y), 1])
    new_test_set_distance_y = np.zeros([len(test_set_distance_y), 1])
    new_train_set_sequence_y = np.zeros([len(train_set_sequence_y), 1])
    new_train_set_time_y = np.zeros([len(train_set_time_y), 1])
    new_train_set_distance_y = np.zeros([len(train_set_distance_y), 1])

    negative_sample = np.zeros([len(new_train_set_sequence_y), num_sample])
    negative_time_sample = np.zeros([len(new_train_set_sequence_y), num_sample])
    negative_distance_sample = np.zeros([len(new_train_set_sequence_y), num_sample])

    print("Begin the padding process")
    new_train_set_sequence = padding(train_set_sequence_x, train_set_sequence_y, new_train_set_sequence_x,
                                     new_train_set_sequence_y, max_len)
    new_train_set_time = padding(train_set_time_x, train_set_time_y, new_train_set_time_x, new_train_set_time_y,
                                 max_len)
    new_train_set_distance = padding(train_set_distance_x, train_set_distance_y, new_train_set_distance_x,
                                     new_train_set_distance_y, max_len)

    mask_train_x = generate_mask(train_set_sequence_x, max_len)

    new_test_set_sequence = padding(test_set_sequence_x, test_set_sequence_y, new_test_set_sequence_x,
                                    new_test_set_sequence_y, max_len)

    new_test_set_time = padding(test_set_time_x, test_set_time_y, new_test_set_time_x, new_test_set_time_y, max_len)

    new_test_set_distance = padding(test_set_distance_x, test_set_distance_y, new_test_set_distance_x,
                                    new_test_set_distance_y, max_len)

    mask_test_x = generate_mask(test_set_sequence_x, max_len)

    negative_samples, negative_distance_samples = padding_negative_sample_SD(train_set_sequence_x, negative_sample,
                                                                             negative_distance_sample, clusters, top_13,
                                                                             coordinates)

    for i in range(num_sample):
        negative_time_sample[:, i] = train_set_time_y

    vocabulary_distances = SD.padding_vocabulary_distance(test_set_sequence_x, n_classes, coordinates)
    final_train_set = (new_train_set_sequence, new_train_set_time, new_train_set_distance, mask_train_x, train_set_user)
    final_test_set = (new_test_set_sequence, new_test_set_time, new_test_set_distance, mask_test_x, test_set_user)
    final_negative_samples = (negative_samples, negative_time_sample, negative_distance_samples)
    return final_train_set, final_test_set, final_negative_samples, vocabulary_distances


def batch_iter_sample(data, negative_samples, batch_size):
    sequence, time, distance, mask_x, user = data
    negative_sample, negative_time_sample, negative_distance_sample = negative_samples
    sequence_x, sequence_y = sequence
    time_x, time_y = time
    distance_x, distance_y = distance

    data_size = len(sequence_x)

    num_batches_per_epoch = int(data_size / batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_sequence_x = sequence_x[start_index:end_index, :]
        return_sequence_y = sequence_y[start_index:end_index, :]

        return_time_x = time_x[start_index:end_index, :]
        return_time_y = time_y[start_index:end_index, :]

        return_distance_x = distance_x[start_index:end_index, :]
        return_distance_y = distance_y[start_index:end_index, :]

        return_mask_x = mask_x[start_index:end_index, :]

        return_negative_sample = negative_sample[start_index:end_index, :]
        return_negative_time_sample = negative_time_sample[start_index:end_index, :]
        return_negative_distance_sample = negative_distance_sample[start_index:end_index, :]
        return_user = user[start_index:end_index]

        yield (return_sequence_x, return_sequence_y, return_time_x, return_time_y, return_distance_x,
               return_distance_y, return_mask_x, return_negative_sample, return_negative_time_sample,
               return_negative_distance_sample, return_user)



def POI_Capacity(dataset):
    capacity = {}
    dfNodes = pd.read_excel('DataExcelFormat/POI-Capacity-' + dataset + '.xlsx')
    for i in range(len(dfNodes)):
        poiID =  dfNodes.iloc[i].poiID
        cap = dfNodes.iloc[i].capacity

        capacity[poiID] = cap
    return capacity



def convert_user2id(users):
    index = 1
    useduser = []
    user2id = {}
    for user in users:
        if user in useduser:
            continue
        else:
            user2id[user] = index
            index = index + 1
            useduser.append(user)

    new_user_list = [user2id[user] for user in users]

    return new_user_list, index


def social_data(numUsers):
    config = Config()
    eval_config = Config()

    coordinates = dc.findCoordinates(dataset)
    category = dc.findCategory(dataset)
    dfVisits = dc.findCheckinInfo(dataset, 28800, numUsers)
    # print("Cordinates = ", coordinates)
    # print("Category = ", category)
    # print("df Visits = ", dfVisits)



    userlocation = dfVisits[['nsid', 'takenUnix', 'poiID', 'seqID']]
    userlocation.nsid, total_user = convert_user2id(userlocation.nsid)

    userlocation = np.array(userlocation)

    userlocation = userlocation.T

    newsequence, sequence_user, sequence_time, sequence_distance, max_len = _build_sequence_SD(userlocation,
                                                                                               coordinates)

    # print("New Sequence = ", newsequence)
    # print("sequence user = ", sequence_user)
    # print(" sequence time = ", sequence_time)
    # print(" sequence distance = ", sequence_distance)

    max_len = max(max_len, 10)  # at least seqence length is 10

    sequence, clusters = new_build_location_voc_SD(newsequence, coordinates)
    # print("Sequence = ", sequence)
    # print("Clusters = ", clusters)
    sequence_user = np.array(sequence_user)

    sequence_user = np.array(sequence_user)
    max_POIs_IDs = max([max(x) for x in sequence])
    # print("Max pois = ", max_POIs_IDs)
    n_classes = max_POIs_IDs

    top_13 = pop_n(sequence, n_classes)
    # print("Top k = ", top_13)

    config.vocab_size = n_classes + 1

    config.num_sample = max_len
    config.num_steps = max_len

    # total_user = len(set(sequence_user))
    config.user_size = total_user
    eval_config.user_size = total_user

    train_set = (sequence, sequence_user, sequence_time, sequence_distance)

    final_train_set, final_test_set, final_negative_samples, vocabulary_distance = load_data_SD(train_set, coordinates,
                                                                                                config.num_sample,
                                                                                                clusters, top_13,
                                                                                                n_classes, 0.3, max_len,
                                                                                                True)

    new_train_set_sequence, new_train_set_time, new_set_distance, mask_train_x, trains_set_user = final_train_set

    if new_train_set_sequence[0].shape[1] <= 10:
        config.batch_size = 1
    else:
        if new_train_set_sequence[0].shape[1] <= 50:
            config.batch_size = 2
        else:
            if new_train_set_sequence[0].shape[1] <= 100:
                config.batch_size = 5
            else:
                if new_train_set_sequence[0].shape[1] <= 200:
                    config.batch_size = 10
                else:
                    config.batch_size = 20

    print("Begin the training process")

    eval_config.keep_prob = 1.0
    eval_config.num_steps = config.num_steps
    eval_config.vocab_size = config.vocab_size
    eval_config.batch_size = 20

    # size = hidden_size
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-1 * config.init_scale, 1 * config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = AT_LSTM(config=config, is_training=True)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            test_model = AT_LSTM(config=config, is_training=False)

        sumamary_writer = tf.summary.FileWriter('/DataExcelFormat/lstm_logs', session.graph)

        tf.global_variables_initializer().run()

        global_steps = 1
        begin_time = int(time.time())

        for i in range(config.max_max_epoch):
            print("The %d epoch training ...." % (i + 1))
            # lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            # model.assign_new_lr(session, config.learning_rate * lr_decay)
            global_steps, cost = run_epoch(model, session, final_train_set, final_negative_samples, global_steps,
                                           config.batch_size)
            if cost < 0.005:
                break

        print(" The train is finished")
        end_time = int(time.time())

        print("Training takes %d seconds already \n" % (end_time - begin_time))
        pre_5, pre_10, f1_5, f1_10, recall_5, recall_10, ndcg_5, ndcg_10 = evaluate(test_model, session, final_test_set,vocabulary_distance, eval_config.batch_size)
        # print("The test data total_precison 5 is %f, total_precision_10 is %f" % (pre_5, pre_10))
        # print("The test data total_acc_5 or Recall_5 is %f total_acc_10 or Recall_10 is %f" % (recall_5, recall_10))
        # print("The test data total_f1_5 is %f, total_f1_10 is %f" % (f1_5, f1_10))
        # print("The test total_ndcg5 is %f, total_ndcg10 is %f" % (ndcg_5, ndcg_10))

        #print("Program end")
        return pre_5, pre_10, f1_5, f1_10, recall_5, recall_10, ndcg_5, ndcg_10



def pred_next_string2(value,targets):

    # print("value = ", value)
    # print("targets = ", targets)

    totalCount = 0
    index = 0

    total_p5, total_r5, total_f5 = 0.0, 0.0, 0.0

    for i, (v,t) in enumerate(zip(value,targets)):

        v = np.asarray(v)
        t = np.asarray(t)
        target_items = []
        predict_items  = []

        index2 = 0
        for j in range(len(t)):
            if t[j] != 0:
                target_items.append(int(t[j]))
                predict_items.append(v[j])
                index2 = index2 + 1

        if index2 > 0:
            c_pre = (len(set(target_items).intersection(set(predict_items)))/len(set(predict_items)))
            r_pre = (len(set(target_items).intersection(set(predict_items))) / len(set(target_items)))
            f_pre = ((2 * len(set(target_items).intersection(set(predict_items)))/len(set(predict_items)) * len(set(target_items).intersection(set(predict_items))) / len(set(target_items)) ) / (len(set(target_items).intersection(set(predict_items)))/len(set(predict_items)) + len(set(target_items).intersection(set(predict_items))) / len(set(target_items)) + 1e-6))

            # print(c_pre, r_pre, f_pre)
            total_p5 = total_p5 + float(c_pre)
            total_r5 = total_r5 + float(r_pre)
            total_f5 = total_f5 + float(f_pre)

            totalCount = totalCount + 1


            index = index + 1

    # print("index = ", index , " index 2 = ", index2, " totalCount = ",totalCount)

    total_p5 = total_p5 /  totalCount
    total_r5 = total_r5 / totalCount
    total_f5 = total_f5 / totalCount



    return total_p5, total_r5, total_f5 #, total_r10, total_f10 #r_5, f_5,

# Fair Evaluation metrics

# Fair Evaluation metrics


def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    # print (len(predicted), "dcg = ", dcg, idcg)
    return dcg / idcg
''' Added code for FAIR REcommendation '''
def evalutation_metrics(A, targets, topk):

    # print(targets)
    # targets = [i[0] for i in targets] # test_sequence_itinerary]
    #
    # rec_topk_length = len(recommended_topk_list)
    results = []
    for i in range(len(A)):

        original =  np.asarray([int(targets[i][0])])
        predict = np.asarray([int(j) for j in A[i]])

        # print("original = ",original)
        # print("predict = ", predict)

        p = precisionk(original, predict[0:topk])
        r = recallk(original, predict[0:topk])
        f = (2 * p * r) / (p + r + 1e-8)

        ndcg = ndcgk(original, predict[0:topk])

        results.append(np.asarray([p,r,f,ndcg]))

    results = np.asarray(results)

    results_mean = [sum(i) / len(results) for i in zip(*results)]

    return results_mean


def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    # array[0] = -10000.00  # Avoid first index for selection

    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]

    indices = indices[np.argsort(-flat[indices])]

    values = np.sort(np.array(array))[::-1]

    return np.asarray(np.unravel_index(indices, array.shape))[0], values[0:n]

def FairRecModel(value, topk, alpha):
    topk = int(topk)
    prediction_score = []

    topk_items = []
    topk_score = []
    length = 0
    for v in value:
        temp_value = v
        prediction_score.append([i for i in temp_value])

        index_topk, score_topk = largest_indices(v, len(temp_value))
        top_k = np.asarray(index_topk)
        top_k_score = np.asarray(score_topk)

        topk_items.append(top_k)
        topk_score.append(top_k_score)

        length = len(temp_value)

    V = np.asarray(prediction_score)
    #
    # Before_Fair_V = np.asarray(topk_items)
    # Before_Fair_V_values = np.asarray(topk_score)

    m = V.shape[0]  # number of customers
    n = V.shape[1]  # number of producers

    E_bar = int(m*topk/n)

    U = range(m)  # list of customers
    P = range(n)  # list of producers

    # print("U =", U)
    # print("P = ", P)

    # calling FairRec
    A = FairRec(U, P, topk , V, alpha, n, m)
    A_Plus = FairRecPlus(U, P, topk, V, alpha, n, m)
    return A, A_Plus, V, E_bar


# greedy round robin allocation based on a specific ordering of customers
# This is the modified greedy round robin where we remove envy cycles
def greedy_round_robin_Plus(m, n, R, l, T, V, U, F):
    print(m, n, R, l, T, V.shape, len(U))

    # creating empty allocations
    B = {}
    for u in U:
        B[u] = []

    # available number of copies of each producer
    Z = {}  # total availability
    P = range(n)  # set of producers
    for p in P:
        Z[p] = l

    # number of rounds
    r = 0
    while True:
        # number of rounds
        r = r + 1
        # allocating the producers to customers
        print("GRR round number==============================", r)

        for i in range(m):
            # user
            u = U[i]

            # choosing the p_ which is available and also in feasible set for the user
            possible = [(Z[p] > 0) * (p in F[u]) * V[u, p] for p in range(n)]
            p_ = np.argmax(possible)

            if (Z[p_] > 0) and (p_ in F[u]) and len(F[u]) > 0:
                B[u].append(p_)
                F[u] = [i for i in F[u] if i != p_] # F[u].remove(p_)
                Z[p_] = Z[p_] - 1
                T = T - 1

            else:  # stopping criteria
                print("now doing envy cycle removal")
                B, U = remove_envy_cycle(B.copy(), U[:], V)
                return B.copy(), F.copy()

            if T == 0:  # stopping criteria
                print("now doing envy cycle removal")
                B, U = remove_envy_cycle(B.copy(), U[:], V)
                return B.copy(), F.copy()
        # envy-based manipulations, m, U, V, B.copy()
        print("GRR done")

    # remove envy cycle
    print("now doing envy cycle removal")
    B, U = remove_envy_cycle(B.copy(), U[:], V)
    print(sum([len(B[u]) for u in B]), T, n * l)
    # returning the allocation
    return B.copy(), F.copy();


def remove_envy_cycle(B,U,V):
    r=0
    while True:
        r+=1
        # print("In envy cycle removal:",r)
        # create empty graph
        G=nx.DiGraph()
        # add nodes
        G.add_nodes_from(U)
        # edges
        E=[]
        CycleFound = True
        # find edges
        print("In envy cycle removal: finding edges")
        for u in U:
            for v in U:
                if u!=v:
                    V_u=0
                    V_v=0
                    for p in B[u]:
                        V_u+=V[u,p]
                    for p in B[v]:
                        V_v+=V[u,p]
                    if V_v>V_u:
                        E.append((u,v))
        # add edges to the graph
        G.add_edges_from(E)
        # find cycle and remove
        print("In envy cycle removal: graph done, finding and removing cycles")
        try:
            cycle=nx.find_cycle(G,orientation="original")
            # print("cycel len = ")
            temp=B[cycle[0][0]][:]
            for pair in cycle:
                B[pair[0]]=B[pair[1]][:]
            B[cycle[-1][0]]=temp[:]
            print("Cycle found..")
            # CycleFound = False
        except:
            CycleFound = False
            print("No cycle found")

        if not CycleFound:
            break
    # topological sort
    U=list(nx.topological_sort(G))
    return B, U # B.copy(),U [:]

def FairRecPlus(U, P, k, V, alpha, n, m):
    # Allocation set for each customer, initially it is set to empty set
    A = {}
    for u in U:
        A[u] = []

    # feasible set for each customer, initially it is set to P
    F = {}
    for u in U:
        F[u] = P[:]
    # print(sum([len(F[u]) for u in U]))

    # l= number of copies of each producer, equal to the exposure guarantee for producers
    l = int(alpha * m * k / (n + 0.0))

    # R= number of rounds of allocation to be done in first GRR
    R = int(math.ceil((l * n) / (m + 0.0)))

    # T= total number of products to be allocated
    T = l * n

    # first greedy round-robin allocation
    B = {}
    [B, F1] = greedy_round_robin_Plus(m, n, R, l, T, V, U[:], F.copy())
    F = {}
    F = F1.copy()
    print("GRR done")
    # adding the allocation
    for u in U:
        A[u] = A[u][:] + B[u][:]

    # filling the recommendation set upto size k
    u_less = []
    for u in A:
        if len(A[u]) < k:
            u_less.append(u)
    for u in u_less:
        scores = V[u, :]
        new = scores.argsort()[-(k + k):][::-1]
        for p in new:
            if p not in A[u]:
                A[u].append(p)
            if len(A[u]) == k:
                break
    # end_time = datetime.datetime.now()
    A = np.asarray([A[a] for a in A.keys()])
    return A;

def greedy_round_robin(m,n,R,l,T,V,U,F):
  # greedy round robin allocation based on a specific ordering of customers (assuming the ordering is done in the relevance scoring matrix before passing it here)
  # creating empty allocations
  B = {}
  for u in U:
    B[u] = []

  # available number of copies of each producer
  Z = {}  # total availability
  P = range(n)  # set of producers
  for p in P:
    Z[p] = l

  # allocating the producers to customers
  for t in range(1, R + 1):
    print("GRR round number==============================", t)
    for i in range(m):
      if T == 0:
        return B, F
      u = U[i]
      # choosing the p_ which is available and also in feasible set for the user
      possible = [(Z[p] > 0) * (p in F[u]) * V[u, p] for p in range(n)]
      p_ = np.argmax(possible)

      if (Z[p_] > 0) and (p_ in F[u]) and len(F[u]) > 0:
        B[u].append(p_)
        F[u] = [i for i in F[u] if i != p_] #  F[u].remove(p_)
        Z[p_] = Z[p_] - 1
        T = T - 1
      else:
        return B, F
  # returning the allocation
  return B, F


import math


def FairRec(U,P,k,V,alpha, n, m):
    # Allocation set for each customer, initially it is set to empty set
    A = {}
    for u in U:
        A[u] = []
    # print("A = " , A)

    # feasible set for each customer, initially it is set to P
    F = {}
    for u in U:
        F[u] = P[:]
    # print("F = ", F)

    # number of copies of each producer
    l = int(alpha * m * k / (n + 0.0))
    # print("l = ", l)

    # R= number of rounds of allocation to be done in first GRR
    R = int(math.ceil((l * n) / (m + 0.0)))

    # print("R = ", R)

    # print(m,n,l,R)

    # total number of copies to be allocated
    T = l * n

    # print("T = ", T)

    # first greedy round-robin allocation
    [B, F1] = greedy_round_robin(m, n, R, l, T, V, U[:], F.copy())

    # print("B = ", B)
    # print("F1 = ", F1)

    F = {}
    F = F1.copy()

    print("1 GRR done")
    # adding the allocation
    for u in U:
        # temp=A[u][:]
        # temp+=B[u][:]
        # A[u]=temp[:]
        A[u] = A[u][:] + B[u][:]

    # second phase
    u_less = []  # customers allocated with <k products till now
    for u in A:
        if len(A[u]) < k:
            u_less.append(u)

    # allocating every customer till k products
    for u in u_less:
        scores = V[u, :]
        new = scores.argsort()[-(k + k):][::-1]
        for p in new:
            if p not in A[u]:
                A[u].append(p)
            if len(A[u]) == k:
                break

    A = np.asarray([A[a] for a in A.keys()])
    return A

def envyScore(data1, data2,  u1_item, u2_item):
    u1_score1, u1_score2 = 0.0, 0.0
    u2_score1, u2_score2 = 0.0, 0.0
    for i in range(len(u1_item)):
        if u1_item[i] not in u2_item:
            u1_score1 = u1_score1 + data1[int(u1_item[i])]
            u2_score1 = u2_score1 + data2[int(u1_item[i])]
        if u2_item[i] not in u1_item:
            u1_score2 = u1_score2  + data1[int(u2_item[i])]
            u2_score2 = u2_score2 + data2[int(u2_item[i])]


    return max((u1_score2-u1_score1)/len(u1_item), 0), max((u2_score1-u2_score2)/len(u1_item),0)


def user_side_matrics(A, V, topk):

    data = np.asarray([x[0:topk] for x in A])

    Y_value = 0
    Loss_value = 0
    Item_loss_value = []
    for i in range(data.shape[0]):
        score = [V[i][int(j)] for j in data[i] ]
        R_u_score = np.mean(score)
        # print("score = ", score)
        Loss_value = Loss_value + R_u_score
        Item_loss_value.append(R_u_score)

        Envey_score = 0.0
        for j in range(i+1,data.shape[0]):
            if i != j:
                Envey_score1, Envey_score2 = envyScore(V[i], V[j], A[i], A[j])
                Envey_score = Envey_score + Envey_score1 + Envey_score2   # take the average score gap between two users

        Envey_score = Envey_score / (data.shape[0]-i - 1+1e-8)
        Y_value = Y_value + Envey_score

    Y_value = Y_value / data.shape[0]
    Loss_value = Loss_value / data.shape[0]

    Item_loss_value = [k for k in Item_loss_value] # [0.0 if k < 0.0 else k for k in Item_loss_value]
    Item_loss_value = np.asarray(Item_loss_value)

    # Calculate stdev
    Std_value = 0.0
    for i in Item_loss_value:
        Std_value = Std_value + (i - Loss_value) * (i - Loss_value)

    Std_value = Std_value / (len(Item_loss_value) - 1)

    Std_value = math.sqrt(Std_value)

    return Y_value, Loss_value, Std_value

def capculate_Envy_Free_Score(un_Demand_ratio, Envy_cap):
    count_Envy = 0.0
    total_pair = 0.0
    pois = [i for i in un_Demand_ratio.keys()]

    for i in range(len(un_Demand_ratio)-1):
        for j in range(i+1, len(un_Demand_ratio)):
            ratio_diff = abs(un_Demand_ratio[pois[i]] - un_Demand_ratio[pois[j]])
            if ratio_diff >= Envy_cap:
                count_Envy = count_Envy + 1.0
            total_pair = total_pair + 1.0

    return 1.0 - count_Envy/ total_pair



def capacity_based_producer_side_metrics(data, min_under_demand, nodelist , topk, envy_cap,  topk_list, new_capacity):
    # print( "nodelist = ", nodelist )
    m, k = data.shape
    n = len(nodelist)

    # total_Capacity = np.sum([capacity[i] for i in capacity.keys()])
    #
    #
    # ratio = math.ceil(m * DEFINES.topk / total_Capacity)
    # # ratio = math.ceil(U / total_Capacity)
    # new_capacity = {}
    # for i in capacity.keys():
    #     new_capacity[i] = ratio * capacity[i]

    # nodelist = [node + 1 for node in nodelist]
    id, counts= recommended_POI_distributions(data, topk, nodelist)
    # print("ids = ", id)
    # print("counts = ", counts)

    id_before, counts_before = recommended_POI_distributions(topk_list, topk, nodelist)
    # print("id before  = ", id_before)
    # print("counts_before = ", counts_before)
    #
    # print("new capacity = ", new_capacity)
    H_value = 0.0
    Z_value = 0.0
    L_value = 0.0
    un_Demand_ratio = {}


    for i, c in enumerate(counts):
        poi = id[i]   # Add 1 for making id mapping
        # print(poi)
        if c >= new_capacity[poi]:
            H_value = H_value + 1.0

        Z_value = Z_value + c / (m * k * new_capacity[poi] + 1e-8) * math.log(c / (m * k * new_capacity[poi] + 1e-8), n)
        L_value = L_value + max((counts_before[i] - c) / counts_before[i], 0.0)
        un_Demand_ratio[poi] = max((new_capacity[poi] - c)/new_capacity[poi], 0.0)



    H_value = H_value / len(counts)
    Z_value = -Z_value
    L_value = L_value / len(counts)

    Envy_Free_Score = capculate_Envy_Free_Score(un_Demand_ratio, envy_cap, min_under_demand)

    return H_value, Z_value, L_value, Envy_Free_Score
#
# def capacity_based_producer_side_metrics(data, min_under_demand, nodelist ,topk_list, new_capacity, min_capacity, topk, envy_cap, ratio):
#     m, k = data.shape
#     n = len(nodelist)
#     id, counts= recommended_POI_distributions(data,topk, nodelist)
#
#     id_before, counts_before = recommended_POI_distributions(topk_list, topk, nodelist)
#
#
#     cFSP_value = 0
#     iPE_value = 0
#     eLP_value = 0
#     un_Demand_ratio = {}
#     for i, c in enumerate(counts):
#         poi = id[i]
#         if c >= min_capacity[poi] :
#             cFSP_value = cFSP_value + 1
#
#         iPE_value = iPE_value + c / (m * k * new_capacity[poi]/ratio) * math.log(c / (m * k * new_capacity[poi]/ratio), n)
#
#         eLP_value = eLP_value + max((counts_before[i] - c) / counts_before[i], 0)
#
#         un_Demand_ratio[poi] = max((new_capacity[poi] - c)/new_capacity[poi], 0.0)
#
#
#
#     cFSP_value = cFSP_value / len(counts)
#     iPE_value = -iPE_value
#     eLP_value = eLP_value / len(counts)
#
#     #print("un_Demand_ratio = ", un_Demand_ratio)
#     Envy_Free_Score = capculate_Envy_Free_Score(un_Demand_ratio, envy_cap)
#
#     return cFSP_value, iPE_value, eLP_value, Envy_Free_Score


def producer_side_metrics(data, E_bar, nodelist ,topk_list, topk):

    id, counts= recommended_POI_distributions(data, topk, nodelist)

    id_before, counts_before = recommended_POI_distributions(topk_list, topk, nodelist)

    m, k = data.shape
    n = len(nodelist)
    H_value = 0
    Z_value = 0
    L_value = 0
    for c in range(len(counts)):
        if counts[c] >= E_bar:
            H_value = H_value + 1

        Z_value = Z_value + counts[c] / (m * k) * math.log(counts[c] / (m * k), n)

        L_value = L_value + max((counts_before[c] - counts[c]) / counts_before[c], 0)

    H_value = H_value / len(counts)
    Z_value = -Z_value
    L_value = L_value / len(counts)

    return H_value, Z_value, L_value

def recommended_POI_Capacity(A, max_len):
    print("A = ", A)
    items = np.array(A).reshape(len(A)* max_len)
    unique, counts = np.unique(items, return_counts=True)

    return unique, counts

def recommended_POI_distributions(A, topk, Nodelist):
    items = [[i for i in y[0:min(topk, A.shape[1])]] for y in A]
    items = np.array(items).reshape(len(items) * min(topk, len(Nodelist)))
    unique, counts = np.unique(items, return_counts=True)

    new_unique = []
    new_counts = []
    for i in Nodelist:
        if i in unique:
            new_unique.append(i)
            new_counts.append(counts[np.where(unique == i)][0])
        else:

            new_unique.append(i)
            new_counts.append(1)

    return new_unique, new_counts

def Least_Significant_Users(tempTOPK_A, tempTOPK_V, p):
    index = np.where(tempTOPK_A == p)
    value = tempTOPK_V[index]
    min_index = (value).argsort()[0:1]
    return min_index

def random_and_mixed(V, topk):
    # Sort based on values decreasing order
    TOP_A = (-V).argsort()

    #Select top k randomly
    random_A = np.asarray([np.random.choice(item, topk) for item in TOP_A])

    # Select first half based on topk/2 items
    half_TOP_K = TOP_A[:,0:topk//2]

    # Select last half based on random order
    half_Random_K = np.asarray([np.random.choice(item[topk//2:], topk - topk//2) for item in TOP_A])

    # Marge these two and makes mixed items
    Mixed_A =  np.concatenate((half_TOP_K,half_Random_K), axis = 1)

    return random_A, Mixed_A



def update_POI_Capactity(C, topk, U, min_dim):
    # Remove first 0-3 index based POI capacity because that are used for special purpouse rather than POI number
    # for i in range(4):
    #     pop_value = C.pop(i, None)
    # print("C = ", C)

    new_capacity = C  # {}
    minimum_capacity = {}
    # for i in C.keys():
    #     if i > 3:
    #         new_capacity[i] = C[i]

    total_Capacity = np.sum([new_capacity[i] for i in new_capacity.keys()])
    ratio = math.ceil(U * topk / total_Capacity)

    for i in new_capacity.keys():
        new_capacity[i] = int(ratio * C[i])
        minimum_capacity[i] = int(new_capacity[i] * (1.0-min_dim))

    return new_capacity, minimum_capacity, ratio



def Fair_FTPR_Model(V, topk, new_capacity, min_capacity):
    # Number of Users and POIs
    U, N = V.shape

    # Sort POIs and valus based on values decreasing order
    TOPK_A = (-V).argsort()  # [:, 0:topk]
    TOPK_V = np.asarray([[V[j, TOPK_A[j, i]] for i in range(len(TOPK_A[j]))] for j in range(TOPK_A.shape[0])])

    usersID = range(TOPK_A.shape[0])
    # print(POINumber)
    Users_POIs = {}
    POIs_Users = {}
    # Initialize users
    for i in usersID:
        Users_POIs[i] = []

    # new_total_capacity = np.sum([new_capacity[i] for i in new_capacity.keys()])
    # print("new_total_capacity = ", new_total_capacity)
    # print("new_capacity 2= ", new_capacity.keys())
    # # print(Users_POIs)
    # print(POIs_Users)


    Demand_POIs = np.asarray([i[0:topk] for i in TOPK_A])  # [:, 0:topk]]  # Find top 1, top2,top3 and so no list
    Demand_POIs_Values = np.asarray([i[0:topk] for i in TOPK_V])  # [:, 0:topk]])  # Find top 1, top2,top3 values and so no list
    # print("POI demands = ", Demand_POIs)
    # print("POI deamnds value = ", Demand_POIs_Values)

    POIs, Demands = recommended_POI_distributions(Demand_POIs, topk, new_capacity.keys())

    # Sorted POIs based on Demands

    enumerate_object = enumerate(Demands)
    sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1), reverse=True)
    sorted_indices = [index for index, element in sorted_pairs]
    #
    # print("POIs = ", POIs)
    # print("Demaands = ", Demands)
    # print("sorted indices = ", sorted_indices)

    POIs = [POIs[i] for i in sorted_indices]
    Demands = [Demands[i] for i in sorted_indices]

    # print("POIs = ", POIs)
    # print("Demaands = ", Demands)

    Over_Demend_POIs = []
    Under_Demand_POIs = []
    for index, value in enumerate(POIs):
        if Demands[index] > new_capacity[value]:
            Over_Demend_POIs.append(value)

        else:
            Under_Demand_POIs.append(value)

    # print ("Over demanded POIs ", Over_Demend_POIs)
    # print ("Under Demanded POI ", Under_Demand_POIs)

    # print("Users = ", range(U))

    Recommended_POIs_to_User = {}
    Assign_Users_to_POI = {}

    Minimum_Allocated_POIs = []
    Exact_Assigend_Users = []

    # Initially users and POIs are empty

    for poi in set(new_capacity.keys()):
        Assign_Users_to_POI[poi] = []

    for user in range(U):
        Recommended_POIs_to_User[user] = []

    # Find Assign Users for a particular POI
    for POI in Over_Demend_POIs:
        find_POI_allocation_index = np.where(Demand_POIs == POI)  # Find where 2 exists

        row, col = find_POI_allocation_index

        # Assign Users Interest Score
        users_interest_value = Demand_POIs_Values[find_POI_allocation_index]

        # Sorted users_interest based list

        sorted_value_index = (-users_interest_value).argsort()
        interested_Users_List = row[sorted_value_index]

        # print("interested_Users_List = ", interested_Users_List)

        # Allocate POIs to the highest interested Users

        for u in interested_Users_List:
            if len(Assign_Users_to_POI[POI]) >= min_capacity[POI]:
                break
            if len(Recommended_POIs_to_User[u]) < topk:
                Assign_Users_to_POI[POI].append(u)  # = Assign_Users_to_POI[POI] + [u]
                Recommended_POIs_to_User[u].append(POI)  # = Recommended_POIs_to_User[u] + [POI]
            else:
                Exact_Assigend_Users.append(u)

        Minimum_Allocated_POIs.append(POI)

    for user in Recommended_POIs_to_User.keys():
        # User personal preferences list
        user_preference_list = [item for item in TOPK_A[user] if item > 3]
        # print(" user_preference_list = ", user_preference_list)
        for poi in user_preference_list:
            # if users get top k POIs than break
            if len(Recommended_POIs_to_User[user]) == topk:
                Exact_Assigend_Users.append(user)
                break


            # Assign POIs if it is underdemanded POIs
            # if poi not in Exact_Allocated_POIs:
            Assign_Users_to_POI[poi].append(user)  # = Assign_Users_to_POI[poi] + [user]
            Recommended_POIs_to_User[user].append(poi)  # [user] = Recommended_POIs_to_User[user] + [poi]
            if len(Assign_Users_to_POI[poi]) == new_capacity[poi]:
                Minimum_Allocated_POIs.append(poi)
                # print("Exact_Allocated_POIs = ", set(Exact_Allocated_POIs))


    for user in Recommended_POIs_to_User.keys():
        if user not in Exact_Assigend_Users:
            user_preference_list = [item for item in TOPK_A[user] if item > 3]
            for poi in user_preference_list:
                # if users get top k POIs than break
                if len(Recommended_POIs_to_User[user]) == topk:
                    Exact_Assigend_Users.append(user)
                    break
                else:

                    Assign_Users_to_POI[poi].append(user)  # = Assign_Users_to_POI[poi] + [user]
                    Recommended_POIs_to_User[user].append(poi)  # [user] = Recommended_POIs_to_User[user] + [poi]
                    if len(Assign_Users_to_POI[poi]) == new_capacity[poi]:
                        Minimum_Allocated_POIs.append(poi)

    # print(" Unassigned Users = " set(Recommended_POIs_to_User) - set(Exact_Assigend_Users))

    FTPR_A = np.asarray([[i for i in Recommended_POIs_to_User[u]] for u in Recommended_POIs_to_User.keys()])

    FTPR_A = np.asarray([[i for i in y] for y in FTPR_A])

    return FTPR_A


def gini(arr):
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

def main(dataset,dataFrame):



    eval_config = Config()


    iteration_number = 1


    dfVisits = pd.read_excel('DataExcelFormat/userVisits-' + dataset + '-allPOI.xlsx')
    dfNodes = pd.read_excel('DataExcelFormat/costProfCat-' + dataset + '-all.xlsx')
    n_classes = max(max(dfNodes.to), max(dfNodes.from1))
    config.vocab_size = n_classes + 1




    poi_capacity = POI_Capacity(dataset)
    nodelist = range(config.vocab_size)

    userlocation = dfVisits[['nsid', 'takenUnix', 'poiID', 'seqID']]
    userlocation.nsid, total_user = convert_user2id(userlocation.nsid)
    userlocation = np.array(userlocation)
    userlocation = userlocation.T

    newsequence, sequence_user, sequence_time, sequence_distance, max_len, sequence_queue = _build_sequence(userlocation, dfNodes)
    sequence, clusters = new_build_location_voc(newsequence, dfNodes)
    sequence_user = np.array(sequence_user)

    max_len = max(max_len, 10) # At least length is 10
    top_13 = pop_n(sequence, n_classes)
    config.num_sample = max_len
    config.num_steps = max_len
    config.user_size = total_user
    eval_config.user_size = total_user

    # make queue prediction
    if (queue_prediction == 1):
        sequence_time = sequence_queue

    train_set = (sequence, sequence_user, sequence_time, sequence_distance)

    final_train_set, final_test_set, final_negative_samples, vocabulary_distance = load_data(train_set,
                                                                                             config.num_sample,
                                                                                             clusters, top_13,
                                                                                             n_classes, dfNodes, 0.3, max_len,
                                                                                             True)

    new_train_set_sequence, new_train_set_time, new_set_distance, mask_train_x, trains_set_user = final_train_set
    # print("trians set users = ", trains_set_user)
    new_test_set_sequence, new_test_set_time, new_test_distance, mask_test_x, tests_set_user = final_test_set

    nex_test_set_sequence_x, new_test_set_sequence_y = new_test_set_sequence
    nex_train_set_sequence_x, new_train_set_sequence_y = new_train_set_sequence

    # pois, capacity = recommended_POI_Capacity(nex_train_set_sequence_x, max_len)
    #
    # poi_capacity = {}
    # for p in range(len(pois)):
    #     poi_capacity[pois[p]] = capacity[p]
    #
    # print("poi capacity = ", poi_capacity)

    if new_train_set_sequence[0].shape[1] <= 10:
        config.batch_size = 1

    elif new_train_set_sequence[0].shape[1] <= 50:
        config.batch_size = 2
    elif new_train_set_sequence[0].shape[1] <= 100:
        config.batch_size = 5
    elif new_train_set_sequence[0].shape[1] <= 200:
        config.batch_size = 10
    else:
        config.batch_size = 20


    if new_test_set_sequence[0].shape[1] <= 10:
        eval_config.batch_size = 1

    elif new_test_set_sequence[0].shape[1] <= 50:
        eval_config.batch_size = 2
    elif new_test_set_sequence[0].shape[1] <= 100:
        eval_config.batch_size = 5
    elif new_test_set_sequence[0].shape[1] <= 200:
        eval_config.batch_size = 10
    else:
        eval_config.batch_size = 20

    print("Begin the training process")

    eval_config.keep_prob = 0.75
    eval_config.num_steps = config.num_steps
    eval_config.vocab_size = config.vocab_size

    data_results_fair = []
    data_results_topk = []
    data_results_rand = []
    data_results_ftpr = []
    data_results_low = []
    data_results_fair_Plus = []

    for iteration in range(iteration_number):
        print("Iteration = ", iteration + 1)

        # size = hidden_size
        with tf.Graph().as_default(), tf.compat.v1.Session() as session:
            initializer = tf.random_uniform_initializer(-1 * config.init_scale, 1 * config.init_scale)

            with tf.compat.v1.variable_scope("model", reuse=None, initializer=initializer):
                model = AT_LSTM(config=config, is_training=True)

            with tf.compat.v1.variable_scope("model", reuse=True, initializer=initializer):
                test_model = AT_LSTM(config=config, is_training=False)

            tf.compat.v1.global_variables_initializer().run()

            global_steps = 1
            begin_time = int(time.time())
            config.learning_rate = 0.0001
            for i in range(config.max_max_epoch):
                print("The %d epoch training ...." % (i + 1))
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                model.assign_new_lr(session, config.learning_rate * lr_decay)
                global_steps, cost = run_epoch(model, session, final_train_set, final_negative_samples, global_steps,
                                               config.batch_size)
                if cost < 0.005:
                    break

            print(" The train is finished")
            end_time = int(time.time())

            print("Training takes %d seconds already \n" % (end_time - begin_time))
            list_value,  list, c_pre_5, c_pre_10, c_f1_5, c_f1_10, c_recall_5, c_recall_10, c_ndcg_5, c_ndcg_10 = evaluate(test_model, session,
                                                                                              final_test_set,
                                                                                              vocabulary_distance,
                                                                                              eval_config.batch_size)


            topk = config.topk
            list_value = np.asarray(list_value)
            # list_value[list_value < 0] = 0.00001  # Avoid negative interest
            # Fiind Fair A list (|U| * top) and V values (|U| * |P| ) and E_bar value
            Fair_A, Fair_A_Plus, V, E_bar = FairRecModel(list_value, config.topk, config.alpha)

            np.savetxt('data_out/' + dataset + '_recommand.csv', V, delimiter=',')
            np.savetxt('data_out/' + dataset + '_original.csv', new_test_set_sequence_y, delimiter=',')

            scaler = MinMaxScaler()
            # print(scaler)
            Norm_V = scaler.fit_transform(list_value)
            # Norm_list_value = np.asarray([[i / max(y) for i in y] for y in list_value])
            # print(results)

            update_poi_capacity, minimum_capacity, ratio = update_POI_Capactity(poi_capacity, config.topk,
                                                                                len(list_value), config.min_under_demand)

            FTPR_A = Fair_FTPR_Model(list_value, config.topk, minimum_capacity, minimum_capacity)


            # Find low top-k items based on lowest score
            Low_TOPK_A = (list_value).argsort()[:, 0:config.topk]
            # print("Low a = ", Low_TOPK_A)

            TOPK_A= (-list_value).argsort()[:, 0:config.topk]
            LOW_TOPK_A = (list_value).argsort()[:, 0:config.topk]

            TOPK_A = np.asarray([[i  for i in y] for y in TOPK_A])
            LOW_TOPK_A = np.asarray([[i  for i in y] for y in LOW_TOPK_A])

            # print(TOPK)
            # print(LOWK)

            reshape_TOPK = TOPK_A.reshape(TOPK_A.shape[0] * config.topk).T
            # print(reshape_TOPK)
            # print(len(reshape_TOPK), TOPK.shape)

            reshape_LOWK = LOW_TOPK_A.reshape(LOW_TOPK_A.shape[0] * config.topk).T
            # print(len(reshape_LOWK), LOWK.shape)

            gini_TOPK = gini(reshape_TOPK)
            gini_LOWK = gini(reshape_LOWK)


            # print(FTPR_A)
            flat_FTPR_A = FTPR_A.reshape(FTPR_A.shape[0] * topk).T
            # print(flat_FTPR_A)
            gini_FTPR = gini(flat_FTPR_A)
            # print('gini FTPR = ', gini_FTPR)

            Random_TOPK_A, Mixed_A = random_and_mixed(V, config.topk)
            Random_TOPK_A = np.asarray([[i  for i in y] for y in Random_TOPK_A])
            Mixed_A = np.asarray([[i for i in y] for y in Mixed_A])

            flat_Random_TOPK_A = Random_TOPK_A.reshape(Random_TOPK_A.shape[0] * Random_TOPK_A.shape[1]).T
            flat_Mixed_A = Mixed_A.reshape(Mixed_A.shape[0] * Mixed_A.shape[1]).T
            gini_Random = gini(flat_Random_TOPK_A)
            gini_Mixed = gini(flat_Mixed_A)
            # print("gini_Random = ", gini_Random)
            # print("gini Mixed = ", gini_Mixed)

            # Fair_A, V, E_bar = FairRecModel(Norm_V[:, 4:], topk, config.alpha)  # We avoid firt 3 charcters
            Fair_A = np.asarray([[i for i in y] for y in Fair_A])
            Fair_A_Plus = np.asarray([[i for i in y] for y in Fair_A_Plus])

            flat_Fair_A = Fair_A.reshape(Fair_A.shape[0] * topk).T
            flat_Fair_A_Plus = Fair_A_Plus.reshape(Fair_A_Plus.shape[0] * topk).T
            # print(Fair_A)
            gini_Fair = gini(flat_Fair_A)
            gini_Fair_Plus = gini(flat_Fair_A_Plus)
            # print("gini Fair = ",gini_Fair)

            # gini_values = [gini_FTPR, gini_Fair, gini_TOPK, gini_Random, gini_Mixed, gini_LOWK]


            # Find Precision, Recall and F1 score value
            FTOP_results_P_R_F = evalutation_metrics(FTPR_A, new_test_set_sequence_y, config.topk)
            # print("FTOP P-R-F = ",FTOP_results_P_R_F)


            Fair_results_P_R_F =evalutation_metrics(Fair_A, new_test_set_sequence_y, config.topk)
            Fair_results_P_R_F_Plus = evalutation_metrics(Fair_A_Plus, new_test_set_sequence_y, config.topk)
            # print("Fair Evaluation = ", Fair_results_P_R_F)

            TOPK_results_P_R_F = evalutation_metrics(TOPK_A, new_test_set_sequence_y, config.topk)
            # print("TOPK Evaluation = ", TOPK_results_P_R_F)

            Low_TOPK_results_P_R_F = evalutation_metrics(Low_TOPK_A, new_test_set_sequence_y, config.topk)

            Random_TOPK_results_P_R_F = evalutation_metrics(Random_TOPK_A, new_test_set_sequence_y, config.topk)

            Mixed_TOPK_results_P_R_F =evalutation_metrics(Mixed_A, new_test_set_sequence_y, config.topk)

            # Calculate User side evaluation metrics



            FTOP_MaE, FTOP_LuU, FTOP_DuU = user_side_matrics(FTPR_A, Norm_V, config.topk)

            # print(" FTOP User side metrics = ", FTOP_MaE, FTOP_LuU, FTOP_DuU)

            Fair_MaE, Fair_LuU, Fair_DuU = user_side_matrics(Fair_A, Norm_V, config.topk)
            Fair_MaE_Plus, Fair_LuU_Plus, Fair_DuU_Plus = user_side_matrics(Fair_A_Plus, Norm_V, config.topk)
            # print(" Fair User side metrics = ", Fair_MaE, Fair_LuU, Fair_DuU)
            TOPK_MaE, TOPK_LuU, TOPK_DuU = user_side_matrics(TOPK_A, Norm_V, config.topk)
            # print(" TOPK User side metrics = ", TOPK_MaE, TOPK_LuU, TOPK_DuU)

            Low_TOPK_MaE, Low_TOPK_LuU, Low_TOPK_DuU = user_side_matrics(Low_TOPK_A, Norm_V, config.topk)

            Random_TOPK_MaE, Random_TOPK_LuU, Random_TOPK_DuU = user_side_matrics(Random_TOPK_A, Norm_V,config.topk)
            Mixed_TOPK_MaE, Mixed_TOPK_LuU, Mixed_TOPK_DuU = user_side_matrics(Mixed_A, Norm_V, config.topk)


            Fair_FSP, Fair_IpS, Fair_ElP, Fair_Envy_Free_Score = capacity_based_producer_side_metrics(Fair_A,
                                                                                                  config.min_under_demand,
                                                                                                  nodelist, topk, config.envy_cap,
                                                                                                  TOPK_A,
                                                                                                  minimum_capacity)
            #capacity_based_producer_side_metrics(Fair_A,config.min_under_demand,nodelist,TOPK_A,update_poi_capacity, minimum_capacity, config.topk, config.envy_cap, ratio)
            Fair_FSP_Plus, Fair_IpS_Plus, Fair_ElP_Plus, Fair_Envy_Free_Score_Plus = capacity_based_producer_side_metrics(Fair_A_Plus, config.min_under_demand,nodelist, topk, config.envy_cap,TOPK_A,minimum_capacity)

            # print("Capacity based Fair Producer side metrics =", Fair_FSP, Fair_IpS, Fair_ElP, Fair_Envy_Free_Score)
            TOPK_FSP, TOPK_IpS, TOPK_ElP, TOPK_Envy_Free_Score = capacity_based_producer_side_metrics(TOPK_A, config.min_under_demand,nodelist, topk, config.envy_cap,TOPK_A,minimum_capacity)
            #capacity_based_producer_side_metrics(TOPK_A, config.min_under_demand,nodelist,TOPK_A,update_poi_capacity, minimum_capacity, config.topk, config.envy_cap, ratio)

            Low_TOPK_FSP, Low_TOPK_IpS, Low_TOPK_ElP, Low_TOPK_Envy_Free_Score = capacity_based_producer_side_metrics(LOW_TOPK_A, config.min_under_demand,nodelist, topk, config.envy_cap,TOPK_A,minimum_capacity)#capacity_based_producer_side_metrics(Low_TOPK_A,config.min_under_demand,nodelist,TOPK_A, update_poi_capacity, minimum_capacity, config.topk, config.envy_cap,ratio)

            Random_TOPK_FSP, Random_TOPK_IpS, Random_TOPK_ElP, Random_TOPK_Envy_Free_Score = capacity_based_producer_side_metrics(Random_TOPK_A, config.min_under_demand,nodelist, topk, config.envy_cap,TOPK_A,minimum_capacity)# capacity_based_producer_side_metrics(Random_TOPK_A, config.min_under_demand, nodelist, TOPK_A, update_poi_capacity, minimum_capacity, config.topk, config.envy_cap, ratio)

            Mixed_TOPK_FSP, Mixed_TOPK_IpS, Mixed_TOPK_ElP, Mixed_TOPK_Envy_Free_Score = capacity_based_producer_side_metrics(Mixed_A, config.min_under_demand,nodelist, topk, config.envy_cap,TOPK_A,minimum_capacity) #capacity_based_producer_side_metrics(Mixed_A, config.min_under_demand, nodelist, TOPK_A, update_poi_capacity, minimum_capacity, config.topk, config.envy_cap, ratio)

            # print("Capacity based TOPK Producer side metrics =", TOPK_FSP, TOPK_IpS, TOPK_ElP, TOPK_Envy_Free_Score)
            # Calculate Producer side evaluation metrics
            FTOP_FSP, FTOP_IpS, FTOP_ElP, FTOP_Envy_Free_Score = capacity_based_producer_side_metrics(FTPR_A, config.min_under_demand,nodelist, topk, config.envy_cap,TOPK_A,minimum_capacity)
            # capacity_based_producer_side_metrics(FTPR_A,
            #                                                                                           config.min_under_demand,
            #                                                                                           nodelist, TOPK_A,
            #                                                                                           update_poi_capacity, minimum_capacity, config.topk, config.envy_cap, ratio)
            # # print("Capacity based FTPR Producer side metrics =", FTOP_FSP, FTOP_IpS, FTOP_ElP, FTOP_Envy_Free_Score)


            # results_after.insert(i for i in [H_value, Z_value,L_value,Y_value, Loss_value,Std_value])
            FTOP_all_results = FTOP_results_P_R_F + [FTOP_MaE, FTOP_LuU, FTOP_DuU, FTOP_FSP, FTOP_IpS,FTOP_Envy_Free_Score, FTOP_ElP] + [gini_FTPR]
            Fiar_all_results = Fair_results_P_R_F + [Fair_MaE, Fair_LuU, Fair_DuU, Fair_FSP, Fair_IpS,
                                                     Fair_Envy_Free_Score, Fair_ElP] + [gini_Fair]
            Fiar_all_results_Plus = Fair_results_P_R_F_Plus + [Fair_MaE_Plus, Fair_LuU_Plus, Fair_DuU_Plus, Fair_FSP_Plus, Fair_IpS_Plus,
                                                     Fair_Envy_Free_Score_Plus, Fair_ElP_Plus] + [gini_Fair_Plus]

            TOPK_all_results = TOPK_results_P_R_F + [TOPK_MaE, TOPK_LuU, TOPK_DuU, TOPK_FSP, TOPK_IpS,
                                                     TOPK_Envy_Free_Score, TOPK_ElP] + [gini_TOPK]
            Low_TOPK_all_results = Low_TOPK_results_P_R_F + [Low_TOPK_MaE, Low_TOPK_LuU, Low_TOPK_DuU, Low_TOPK_FSP,
                                                             Low_TOPK_IpS, Low_TOPK_Envy_Free_Score, Low_TOPK_ElP] +  [gini_LOWK]
            Random_TOPK_all_results = Random_TOPK_results_P_R_F + [Random_TOPK_MaE, Random_TOPK_LuU, Random_TOPK_DuU,
                                                                   Random_TOPK_FSP, Random_TOPK_IpS,
                                                                   Random_TOPK_Envy_Free_Score, Random_TOPK_ElP] +  [gini_Random]
            Mixed_TOPK_all_results = Mixed_TOPK_results_P_R_F + [Mixed_TOPK_MaE, Mixed_TOPK_LuU, Mixed_TOPK_DuU,
                                                                 Mixed_TOPK_FSP, Mixed_TOPK_IpS,
                                                                 Mixed_TOPK_Envy_Free_Score, Mixed_TOPK_ElP] + [gini_Mixed]


            dataFrame.at[dataFrame.shape[0]] = ["FTOPK", dataset, topk] + FTOP_all_results
            dataFrame.at[dataFrame.shape[0]] = ["FairRecPlus", dataset, topk] + Fiar_all_results_Plus
            dataFrame.at[dataFrame.shape[0]] = ["FairRec", dataset, topk] + Fiar_all_results
            dataFrame.at[dataFrame.shape[0]] = ["TOPK", dataset, topk] + TOPK_all_results
            dataFrame.at[dataFrame.shape[0]] = ["LOW_TOPK", dataset, topk] + Low_TOPK_all_results
            dataFrame.at[dataFrame.shape[0]] = ["Random", dataset, topk] + Random_TOPK_all_results
            dataFrame.at[dataFrame.shape[0]] = ["Mixed", dataset,topk] + Mixed_TOPK_all_results
            # data_results_ftop(FTOP_all_results)
            data_results_fair.append(Fiar_all_results)
            data_results_fair_Plus.append(Fiar_all_results_Plus)
            data_results_topk.append(TOPK_all_results)
            data_results_rand.append(Random_TOPK_all_results)
            data_results_ftpr.append(FTOP_all_results)
            data_results_low.append(Low_TOPK_all_results)

    data_results_fair = np.asarray(data_results_fair)
    data_results_fair_mean = [sum(i) / len(data_results_fair) for i in zip(*data_results_fair)]

    data_results_fair_Plus = np.asarray(data_results_fair_Plus)
    data_results_fair_mean_Plus = [sum(i) / len(data_results_fair_Plus) for i in zip(*data_results_fair_Plus)]

    data_results_topk = np.asarray(data_results_topk)
    data_results_topk_mean = [sum(i) / len(data_results_topk) for i in zip(*data_results_topk)]

    data_results_rand = np.asarray(data_results_rand)
    data_results_rand_mean = [sum(i) / len(data_results_rand) for i in zip(*data_results_rand)]
    #
    data_results_ftpr = np.asarray(data_results_ftpr)
    data_results_ftpr_mean = [sum(i) / len(data_results_rand) for i in zip(*data_results_ftpr)]

    data_results_low = np.asarray(data_results_low)
    data_results_low_mean = [sum(i) / len(data_results_rand) for i in zip(*data_results_low)]

    print(dataset, " Final Results Random = ", data_results_rand_mean)
    print(dataset, " Final Results LOWK= ", data_results_low_mean)
    print(dataset, " Final Results TOPK= ", data_results_topk_mean)
    print(dataset, " Final Results FairRec= ", data_results_fair_mean)
    print(dataset, " Final Results FairRecPlus = ", data_results_fair_mean_Plus)
    print(dataset, " Final Results FTOPK= ", data_results_ftpr_mean)



    return dataFrame



'''
Main function 
'''

if __name__ == "__main__":

    datasets = ["epcot","caliAdv","MagicK","Buda","Toro","Melbourne"] #['Toro','Buda','Melbourne' 'epcot','MagicK','caliAdv'] #
   #  #POIs = [1000,2000,3000,4000,5000,6000]
    POIs = [500,1000,1500,2000,2500,3000]
   # #  POIs = [2000, 2500, 3000]
   #  GCN_Used = [1]

    for topk in [5,10]: #,15,20]:

        config.topk = topk
        columnName = ["Algo", "dataset", "top-k", "precision", "recall", "F1-score", "ndcg", "MaE_value", "LuU_value",
                      "DuU_value", "cFSP_value", "iPE_value", "Envy_free_score", "eLP_value", "gini_index"]


        dataFrame = pd.DataFrame(columns=columnName)


        for data in datasets:
            if data in ['Foursquare','Gowalla', 'Yelp']:
                for nPOIs in POIs:
                    config.number_POIs = nPOIs
                    dataFrame = main(data,dataFrame)
            else:

                dataFrame = main(data,dataFrame)
        dataFrame.to_excel("Results_T/FTPR_baselines_ICDM" + str(config.topk) + "_" + str(Config.min_under_demand) +"_"+str(Config.envy_cap)+ ".xlsx", index=False)

