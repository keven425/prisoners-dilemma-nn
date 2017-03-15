import os
import logging
import numpy as np

logger = logging.getLogger("209_project")


class AbstractAgent(object):
    """Abstracts an agent
    """

    def __init__(self, config):
      self.config = config
      self.running_score = np.zeros(shape=(self.config.batch_size,))

    ###################### Build the model ##############################

    def act(self, sess, episode):
        raise NotImplementedError

    def update(self, session, own_action, opponent_action, episolde, reward):
        self.running_score = reward + self.config.discount * self.running_score

    def end_batch(self, sess, saver, best_score):
        raise NotImplementedError
