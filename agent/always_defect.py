import os
import logging

from abstract_agent import AbstractAgent

logger = logging.getLogger("209_project")


class AlwaysDefectAgent(AbstractAgent):
    """Abstracts an agent
    """

    def __init__(self, config):
      super(AlwaysDefectAgent, self).__init__(config)

    ###################### Build the model ##############################

    def act(self, sess, episode):
        return 1

    def update(self, session, own_action, opponent_action, episolde, reward):
      # update running score
      super(AlwaysDefectAgent, self).update(session, own_action, opponent_action, episolde, reward)

    def end_batch(self, sess, saver, best_score):
        score = self.running_score
        self.running_score = 0.0
        return score, None, None
