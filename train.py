import os
import logging
import sys
import time
import math
from datetime import datetime
import atexit

import tensorflow as tf
import numpy as np

from agent.always_cooperate import AlwaysCooperateAgent
from agent.always_defect import AlwaysDefectAgent
from agent.q_learning_finite import QLearningFiniteAgent
from agent.q_learning_finite_fast import QLearningAgentFiniteFast
from agent.q_learning_infinite import QLearningInfiniteAgent
from agent.tit_for_dat import TitForDatAgent
from environment import Environment
from utils import csv
from utils import summary

logger = logging.getLogger("209_project")



class PreExitSaver():
  def __init__(self, config):
    self.config = config

  def update(self, scores1, scores2, actions1, actions2):
    self.scores1 = scores1
    self.scores2 = scores2
    self.actions1 = actions1
    self.actions2 = actions2

  def save(self):
    print('pre-exit saving')
    # save to csv
    scores_path = os.path.join(self.config.model_output_path, 'scores.csv')
    csv.save_scores(scores_path, (self.scores1, self.scores2))
    actions_path = os.path.join(self.config.model_output_path, 'actions.csv')
    csv.save_actions(actions_path, (self.actions1, self.actions2))

    # plot
    summary.scores(scores_path, config=self.config)
    summary.actions(actions_path, config=self.config)
    summary.markov_matrix(actions_path, config=self.config)
    summary.markov_matrix_prob(actions_path, config=self.config)



def run(config):

  preExitSaver = PreExitSaver(config)
  atexit.register(preExitSaver.save)

  with tf.Graph().as_default():
    logger.info("Building model...", )
    start = time.time()
    with tf.variable_scope("Agent1"):
      # agent1 = QLearningAgent(config)
      # agent1 = QLearningAgentFiniteFast(config)
      agent1 = QLearningInfiniteAgent(config)
    # with tf.variable_scope("Agent2"): # don't share weights
      # agent2 = QLearningAgent(config)
      # agent2 = QLearningAgentFast(config)
      # agent2 = QLearningInfiniteAgent(config)
    # agent2 = AlwaysDefectAgent(config)
    agent2 = TitForDatAgent(config)
    # agent2 = AlwaysCooperateAgent(config)
    logger.info("took %.2f seconds", time.time() - start)
    config.agent_names = [type(agent1).__name__, type(agent2).__name__]
    config.game_name = type(agent1).__name__ + ' vs. ' + type(agent2).__name__

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    # saver = None

    scores1 = [] # one per batch
    scores2 = []
    actions1 = [] # one per episode
    actions2 = []

    with tf.Session() as session:
      if config.model_path:
        logger.info("restoring model: " + config.model_path)
        start = time.time()
        saver.restore(session, config.model_path)
        logger.info("took %.2f seconds", time.time() - start)
      else:
        session.run(init)

      # start game/tournament
      best_score = 0.
      for batch_i in range(config.n_batches):

        logger.info("batch %d/%d", batch_i, config.n_batches)
        environment = Environment(config)

        for episode_i in range(config.n_episodes):

          a1 = agent1.act(session, episode_i)
          a2 = agent2.act(session, episode_i)

          _, r1, r2 = environment.step(a1, a2)

          agent1.update(session, a1, a2, episode_i, r1)
          agent2.update(session, a2, a1, episode_i, r2)

          tf.get_variable_scope().reuse_variables()

          actions1.append(a1)
          actions2.append(a2)

        score1, loss1, lr1 = agent1.end_batch(session, saver, best_score)
        score2, loss2, lr2 = agent2.end_batch(session, saver, best_score)
        print('agent1 score: ' + str(score1) + ', loss: ' + str(loss1) + ', lr: ' + str(lr1))
        print('agent2 score: ' + str(score2) + ', loss: ' + str(loss2) + ', lr: ' + str(lr2))

        scores1.append(score1)
        scores2.append(score2)

        #     prog = Progbar(target=1 + int(len(train_padded) / self.config.batch_size))
        #     for i, batch in enumerate(minibatches(train_padded, self.config.batch_size)):
        #         loss, grad_norm, learning_rate = self.train_on_batch(sess, *batch)
        #         prog.update(i + 1, None, exact=[("loss", loss), ('lr', learning_rate), ('gradnorm', grad_norm)])
        #         if self.report: self.report.log_train_loss(loss)

        max_score = np.max([score1, score2])
        if max_score > best_score:
          best_score = max_score

        preExitSaver.update(scores1, scores2, actions1, actions2)

    preExitSaver.save()
