import os
import logging

from abstract_agent import AbstractAgent

logger = logging.getLogger("209_project")


class AlwaysCooperateAgent(AbstractAgent):
    """Abstracts an agent
    """

    def __init__(self, config):
      self.__class__.__name__ = 'Coop'
      super(AlwaysCooperateAgent, self).__init__(config)

    ###################### Build the model ##############################

    def act(self, sess, episode):
        return 0

    def update(self, session, own_action, opponent_action, episolde, reward):
      # update running score
      super(AlwaysCooperateAgent, self).update(session, own_action, opponent_action, episolde, reward)

    def end_batch(self, sess, saver, best_score):
        score = self.running_score
        self.running_score = 0.0
        return score, None, None
