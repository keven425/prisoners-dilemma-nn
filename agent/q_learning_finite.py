import os
import tensorflow as tf
import numpy as np
import logging

from abstract_agent import AbstractAgent

logger = logging.getLogger("209_project")



class QLearningFiniteAgent(AbstractAgent):

  def __init__(self, config):
    super(QLearningFiniteAgent, self).__init__(config)
    # opponent's history of actions
    # actions: 0 = cooperate, 1 = defect, 2 = unknown
    # initialize with unknowns
    self.own_actions = self.empty_actions_history()
    self.opponent_actions = self.empty_actions_history()
    # temporary table for target-Q
    self.Q_target = np.zeros(shape=(config.n_episodes, 2))

    self.build()

  def empty_actions_history(self):
    return np.zeros(shape=(self.config.n_episodes,), dtype=np.int32)
    # return np.ones(shape=(self.config.n_episodes,), dtype=np.int32) * 2

  def add_placeholders(self):
    self.episode_placeholder = tf.placeholder(tf.int32, shape=(None))
    self.states_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_episodes, 2 * self.config.n_episodes))
    self.q_target_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_episodes, 2))

  def create_feed_dict(self, episode, states, q_target=None):
    feed_dict = {
      self.episode_placeholder: episode,
      self.states_placeholder: states
    }
    if not q_target is None:
      feed_dict[self.q_target_placeholder] = q_target
    return feed_dict

  def add_logit_op(self):
    with tf.variable_scope("QLearningAgent"):
      cell = tf.contrib.rnn.LSTMCell(self.config.state_size, num_proj=2)
      # if episode == 0, needs sequence_length to be 1
      sequence_length = self.episode_placeholder + 1
      Q_out, Q_last = tf.nn.dynamic_rnn(cell, self.states_placeholder, sequence_length=sequence_length, dtype=tf.float32)
      Q_last = Q_last[1]
    return Q_out, Q_last

  def add_prediction_op(self, logit, Q_last):
    # Q_last = logit[:, self.episode_placeholder]
    action = tf.argmax(Q_last, axis=1)
    return action

  def add_loss_op(self, logit):
    loss = tf.reduce_mean(tf.square(logit - self.q_target_placeholder))
    return loss

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
    # convert to one-hot vector
    # episode_states = np.identity(self.config.n_episodes)[:episode + 1]
    # states[opponent_action_history] = episode_states

    states = self.get_states(episode)

    feed = self.create_feed_dict([episode], [states])
    action = sess.run(self.pred, feed_dict=feed)
    action = action[0]

    # randomly choose strategy w/ probability
    if np.random.rand(1) < self.config.e:
      action = np.random.randint(2)
    return action

  def get_states(self, episode):
    states = np.zeros(shape=(self.config.n_episodes, 2, self.config.n_episodes), dtype=np.float32)
    opponent_action_history = self.opponent_actions[:episode + 1]
    episode_is = range(episode + 1)
    states[episode_is, opponent_action_history, episode_is] = 1.0
    states = states.reshape((self.config.n_episodes, -1))
    return states

  def update(self, session, own_action, opponent_action, episode, reward):
    # update running score
    super(QLearningFiniteAgent, self).update(session, own_action, opponent_action, episode, reward)
    # update actions history
    self.opponent_actions[episode] = opponent_action
    self.own_actions[episode] = own_action
    # update Q_target
    if (episode + 1) >= self.config.n_episodes:
      # if termination, reward zero
      Q_next = 0.0
    else:
      Q_next = np.max(self.Q_target[episode + 1])
    target = reward + self.config.discount * Q_next
    self.Q_target[episode, own_action] = (1 - self.config.learning_rate) * self.Q_target[episode, own_action] + \
                                      self.config.learning_rate * target

  def end_batch(self, sess, saver, best_score):
    # feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
    # _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
    # return loss

    # print stats
    print('own_actions:')
    print(str(self.own_actions))
    print('Q_target:')
    print(str(self.Q_target))

    # store model
    score = self.running_score
    if score > best_score:
      if saver:
        logger.info("New best score! Saving model in %s", self.config.model_output)
        dirpath = os.path.dirname(self.config.model_output)
        if not os.path.exists(dirpath):
          os.makedirs(dirpath)
        saver.save(sess, self.config.model_output)
    logger.info("")

    # train
    states = self.get_states(self.config.n_episodes - 1)
    feed = self.create_feed_dict([self.config.n_episodes], [states], [self.Q_target])
    loss, _, lr = sess.run([self.loss, self.train_op, self.learning_rate], feed_dict=feed)

    # clear game buffer
    self.running_score = 0.0
    self.opponent_actions = self.empty_actions_history()
    self.own_actions = self.empty_actions_history()

    return score, loss, lr