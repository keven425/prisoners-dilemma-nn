import os
import logging

from abstract_agent import AbstractAgent

logger = logging.getLogger("209_project")


class TitForDatAgent(AbstractAgent):
    """
    repeat opponent's last action
    """

    def __init__(self, config):
      super(TitForDatAgent, self).__init__(config)
      # start off cooperating
      self.opponent_last_action = 0

    ###################### Build the model ##############################

    def act(self, sess, episode):
        return self.opponent_last_action

    def update(self, session, own_action, opponent_action, episolde, reward):
      # update running score
      super(TitForDatAgent, self).update(session, own_action, opponent_action, episolde, reward)
      self.opponent_last_action = opponent_action

    def end_batch(self, sess, saver, best_score):
        score = self.running_score
        self.running_score = 0.0
        return score, None, None
