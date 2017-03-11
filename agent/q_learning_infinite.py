import os
import tensorflow as tf
import numpy as np
import logging

from abstract_agent import AbstractAgent

logger = logging.getLogger("209_project")



class QLearningInfiniteAgent(AbstractAgent):

  def __init__(self, config, n_layer=1):
    self.name = 'Q-' + str(n_layer) + 'layer'
    self.n_layer = n_layer
    super(QLearningInfiniteAgent, self).__init__(config)
    self.total_episode = 0
    # opponent's history of actions
    # actions: 0 = cooperate, 1 = defect
    # initialize with unknowns
    self.own_actions = self.empty_actions_history()
    self.opponent_actions = self.empty_actions_history()
    # temporary table for target-Q. shape = (N prev self actions, N prev opponent actions, N current own action)
    self.Q_target = np.zeros(shape=(2, 2, 2))

    self.build()

  def empty_actions_history(self):
    return np.zeros(shape=(0,), dtype=np.int32)

  def add_placeholders(self):
    # n episodes to look back / feed into RNN
    self.n_episode_lookback_placeholder = tf.placeholder(tf.int32, shape=(None))
    self.states_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_episodes, 2 * 2))
    self.q_target_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_episodes, 2))

  def create_feed_dict(self, n_episode_lookback, states, q_target=None):
    feed_dict = {
      self.n_episode_lookback_placeholder: n_episode_lookback,
      self.states_placeholder: states
    }
    if not q_target is None:
      feed_dict[self.q_target_placeholder] = q_target
    return feed_dict

  def add_logit_op(self):
    with tf.variable_scope("QLearningAgent"):
      cells = []
      if self.n_layer == 2: # only support 1 or 2 layer for now
        _cell = tf.contrib.rnn.LSTMCell(self.config.state_size)
        cells.append(_cell)
      _cell = tf.contrib.rnn.LSTMCell(self.config.state_size, num_proj=2)
      cells.append(_cell)
      cell = tf.contrib.rnn.MultiRNNCell(cells)

      # if episode == 0, needs sequence_length to be 1
      sequence_length = self.n_episode_lookback_placeholder + 1
      Q_out, Q_last = tf.nn.dynamic_rnn(cell, self.states_placeholder, sequence_length=sequence_length, dtype=tf.float32)
      if self.n_layer <= 1:
        Q_last = Q_last[0][1]
      elif self.n_layer == 2:
        Q_last = Q_last[1][1]
    return Q_out, Q_last

  def add_prediction_op(self, logit, Q_last):
    # Q_last = logit[:, self.episode_placeholder]
    # Q_target = Q_last
    action = tf.argmax(Q_last, axis=1)
    return action

  def add_loss_op(self, logit):
    # logit = tf.reduce_mean(logit, axis=1)
    # Q_target = self.get_Q(logit)
    loss = tf.reduce_mean(tf.square(logit - self.q_target_placeholder))
    return loss

  # def get_Q(self, logit):
  #   return tf.reshape(logit, (-1, 2, 2, 2))

  def add_training_op(self, loss):
    game_i = tf.Variable(0, trainable=False)
    self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, game_i, self.config.n_episodes, self.config.lr_decay,
                                                    staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    grads_vars = optimizer.compute_gradients(loss)
    grads = [pair[0] for pair in grads_vars]
    self.grad_norm = tf.global_norm(grads)
    train_op = optimizer.apply_gradients(grads_vars, global_step=game_i)

    return train_op

  def build(self):
    self.add_placeholders()
    self.logit, self.Q_last = self.add_logit_op()
    self.pred = self.add_prediction_op(self.logit, self.Q_last)
    self.loss = self.add_loss_op(self.logit)
    self.train_op = self.add_training_op(self.loss)

  def act(self, sess, episode):
    states, n_episode_lookback = self.get_feed(self.total_episode)
    feed = self.create_feed_dict([n_episode_lookback], [states])
    action = sess.run(self.pred, feed_dict=feed)
    action = action[0]
    assert (self.total_episode % self.config.n_episodes) == episode
    self.total_episode += 1

    # randomly choose strategy w/ probability
    if np.random.rand(1) < self.config.e:
      action = np.random.randint(2)
    return action

  def get_feed(self, total_episode):
    states = np.zeros(shape=(self.config.n_episodes, 2 * 2), dtype=np.float32)
    if total_episode == 0:
      # if beginning of game, assume prev action is 0 for both opponent and self
      return states, 0

    # if in first batch
    opponent_action_history = self.opponent_actions[-self.config.n_episodes:]
    own_action_history = self.own_actions[-self.config.n_episodes:]
    n_episode_lookback = np.min([self.config.n_episodes, total_episode])
    episode_is = range(n_episode_lookback)
    # set corresponding one-hot vector
    states[episode_is, own_action_history] = 1.0
    states[episode_is, opponent_action_history + 2] = 1.0
    return states, n_episode_lookback

  def update(self, session, own_action, opponent_action, episode, reward):
    # update running score
    super(QLearningInfiniteAgent, self).update(session, own_action, opponent_action, episode, reward)
    self.opponent_actions = np.append(self.opponent_actions, opponent_action)
    self.own_actions = np.append(self.own_actions, own_action)
    assert ((self.opponent_actions.shape[0] - episode - 1) % self.config.n_episodes) == 0
    assert ((self.own_actions.shape[0] - episode - 1) % self.config.n_episodes) == 0
    # skip 1st episode, because there're no prev actions
    if episode == 0:
      return
    # update actions history
    prev_own_action = self.own_actions[-2]
    prev_opponent_action = self.opponent_actions[-2]
    # update Q_target
    Q_next = np.max(self.Q_target[own_action][opponent_action])
    target = reward + self.config.discount * Q_next
    self.Q_target[prev_own_action][prev_opponent_action][own_action] = \
            (1 - self.config.learning_rate) * self.Q_target[prev_own_action][prev_opponent_action][own_action] + \
            self.config.learning_rate * target

  def end_batch(self, sess, saver, best_score):

    # store model
    score = self.running_score
    # if score > best_score:
    #   if saver:
    #     logger.info("New best score! Saving model in %s", self.config.model_output)
    #     dirpath = os.path.dirname(self.config.model_output)
    #     if not os.path.exists(dirpath):
    #       os.makedirs(dirpath)
    #     saver.save(sess, self.config.model_output)
    # logger.info("")

    # only keep actions from last n_episodes
    # clear first, so that code later can access trimmed actions history
    self.opponent_actions = self.opponent_actions[-self.config.n_episodes:]
    self.own_actions = self.own_actions[-self.config.n_episodes:]
    # clear game buffer
    self.running_score = 0.0

    # print stats
    print('own_actions:')
    print(str(self.own_actions))
    print('Q_target:')
    print(str(self.Q_target))

    # train
    states, _ = self.get_feed(self.total_episode)
    Q_targets = self.Q_target[self.own_actions, self.opponent_actions]
    feed = self.create_feed_dict([self.config.n_episodes], [states], [Q_targets])
    loss, _, lr = sess.run([self.loss, self.train_op, self.learning_rate], feed_dict=feed)

    return score, loss, lr