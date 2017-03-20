import os
import logging
import numpy as np

from abstract_agent import AbstractAgent

logger = logging.getLogger("209_project")


class ExtortingAgent(AbstractAgent):
    """
    Agent that extorts the other player
    http://www.pnas.org/content/109/26/10409.full.pdf
    markov p = (11/13, 1/2, 7/26, 0)
    best score self = 3.73
    score opponent = 1.91
    """

    def __init__(self, config):
      self.name = 'Extort'
      super(ExtortingAgent, self).__init__(config)
      # keep memory of last actions
      self.prev_self_action = 0
      self.prev_oppo_action = 0
      # here prob is probability of defect. hence 1 - p above
      self.markov_prob = [
        [2.0/13, 1.0/2],
        [19.0/26, 1.0]
      ]

    ###################### Build the model ##############################

    def act(self, sess, episode):
      prob = self.markov_prob[self.prev_self_action][self.prev_oppo_action]
      if np.random.rand() < prob:
        return 1
      else:
        return 0

    def update(self, session, own_action, opponent_action, episolde, reward):
      self.prev_self_action = own_action
      self.prev_oppo_action = opponent_action
      # update running score
      super(ExtortingAgent, self).update(session, own_action, opponent_action, episolde, reward)

    def end_batch(self, sess, saver, best_score):
        score = self.running_score
        self.running_score = 0.0
        return score, None, None