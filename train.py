import os
import logging
import sys
import time
import math
from datetime import datetime
import atexit
import itertools

import tensorflow as tf
import numpy as np

from agent.always_cooperate import AlwaysCooperateAgent
from agent.always_defect import AlwaysDefectAgent
from agent.q_learning_finite import QLearningFiniteAgent
from agent.q_learning_finite_fast import QLearningAgentFiniteFast
from agent.q_learning_infinite import QLearningInfiniteAgent
from agent.extort_agent import ExtortingAgent
from agent.tit_for_dat import TitForDatAgent
from environment import Environment
from utils import csv
from utils import summary

logger = logging.getLogger("209_project")



class PreExitSaver():
  def __init__(self, config):
    self.config = config

  def update(self, scores, actions, actions_pair):
    self.scores = scores
    self.actions = actions
    self.actions_pair = actions_pair

  def save(self, agents):
    print('pre-exit saving')
    # save to csv
    scores_path = os.path.join(self.config.model_output_path, 'scores.npz')
    np.savez(scores_path, self.scores)
    actions_path = os.path.join(self.config.model_output_path, 'actions.npz')
    np.savez(actions_path, self.actions)
    # pairwise actions csv, for computing markov matrix
    actions_pair_path = os.path.join(self.config.model_output_path, 'actions_pair.npz')
    np.savez(actions_pair_path, self.actions_pair)

    # plot
    summary.scores(scores_path, config=self.config)
    summary.actions(actions_path, config=self.config)
    summary.markov_matrix(actions_pair_path, config=self.config)
    summary.markov_matrix_prob(actions_pair_path, config=self.config)

    # print each agents' last words
    log = ''
    for agent in agents:
      log += 'agent: ' + agent.name + ':\n' + \
             agent.log() + '\n\n\n'
    summary.agent_log(log, actions_pair_path)




def run(config):

  preExitSaver = PreExitSaver(config)
  # atexit.register(preExitSaver.save)

  with tf.Graph().as_default():
    logger.info("Building model...", )
    start = time.time()
    agents = []
    for i in range(config.n_q_agents):
      with tf.variable_scope("Agent" + str(i)): # different scope for each agent
        agent = QLearningInfiniteAgent(config)
        agents.append(agent)

    for i in range(config.n_q2_agents):
      with tf.variable_scope("Agent_2l" + str(i)): # different scope for each agent
        agent = QLearningInfiniteAgent(config, n_layer=2)
        agents.append(agent)

    for i in range(config.n_e_agents):
      agent = ExtortingAgent(config)
      agents.append(agent)

    for i in range(config.n_titdat_agents):
      agent = TitForDatAgent(config)
      agents.append(agent)

    for i in range(config.n_c_agents):
      agent = AlwaysCooperateAgent(config)
      agents.append(agent)

    for i in range(config.n_d_agents):
      agent = AlwaysDefectAgent(config)
      agents.append(agent)

    logger.info("took %.2f seconds", time.time() - start)
    config.agent_names = [agent.name for agent in agents]

    init = tf.global_variables_initializer()
    # saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver = None

    scores = [[] for _ in range(config.n_agents)] # one per batch
    actions = [[] for _ in range(config.n_agents)] # one per episode
    action_pairs = [[] for _ in range(config.n_agents)] # shape = n_agent x n_agent x 2
    for i in range(config.n_agents):
      action_pairs[i] = [[[], []] for _ in range(config.n_agents)]

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

        # round robin between pair-combination agents
        for i, j in itertools.combinations(range(config.n_agents), 2):
          agent1 = agents[i]
          agent2 = agents[j]

          logger.info("batch %d/%d", batch_i, config.n_batches)
          environment = Environment(config)

          for episode_i in range(config.n_episodes):

            a1 = agent1.act(session, episode_i)
            a2 = agent2.act(session, episode_i)

            _, r1, r2 = environment.step(a1, a2)

            agent1.update(session, a1, a2, episode_i, r1)
            agent2.update(session, a2, a1, episode_i, r2)

            tf.get_variable_scope().reuse_variables()

            actions[i].append(a1)
            actions[j].append(a2)
            action_pairs[i][j][0].append(a1)
            action_pairs[i][j][1].append(a2)
            action_pairs[j][i][0].append(a2)
            action_pairs[j][i][1].append(a1)

          score1, loss1, lr1 = agent1.end_batch(session, saver, best_score)
          score2, loss2, lr2 = agent2.end_batch(session, saver, best_score)
          logger.info('agent1 score: ' + str(score1) + ', loss: ' + str(loss1) + ', lr: ' + str(lr1))
          logger.info('agent2 score: ' + str(score2) + ', loss: ' + str(loss2) + ', lr: ' + str(lr2))

          scores[i].append(score1)
          scores[j].append(score2)

          #     prog = Progbar(target=1 + int(len(train_padded) / self.config.batch_size))
          #     for i, batch in enumerate(minibatches(train_padded, self.config.batch_size)):
          #         loss, grad_norm, learning_rate = self.train_on_batch(sess, *batch)
          #         prog.update(i + 1, None, exact=[("loss", loss), ('lr', learning_rate), ('gradnorm', grad_norm)])
          #         if self.report: self.report.log_train_loss(loss)

          max_score = np.max([score1, score2])
          if max_score > best_score:
            best_score = max_score

        # only update after one round, where all player pairs played
        preExitSaver.update(scores, actions, action_pairs)

    preExitSaver.save(agents)
