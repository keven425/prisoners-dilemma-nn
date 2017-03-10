import os
import tensorflow as tf
import numpy as np
import argparse
import logging
import time
from datetime import datetime
import pprint
import train

import logging
logging.basicConfig(level=logging.INFO)



tf.app.flags.DEFINE_string("name", "", "Name of this run. will be in model output path")
tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.app.flags.DEFINE_float("lr_decay", 0.9995, "Learning rate decay.")
tf.app.flags.DEFINE_integer("n_episodes", 10, "Number of episodes within a batch")
tf.app.flags.DEFINE_integer("n_batches", 100, "Number of batches to train.")
tf.app.flags.DEFINE_float("discount", 0.95, "Reward discount.")
tf.app.flags.DEFINE_float("e", 0.2, "Epsilon probability to take random action.")
tf.app.flags.DEFINE_integer("n_q_agents", 0, "Number of Q-learning agents in tournament.")
tf.app.flags.DEFINE_integer("n_titdat_agents", 0, "Number of tit-for-dat agents in tournament.")
tf.app.flags.DEFINE_integer("n_c_agents", 0, "Number of always cooperating agents in tournament.")
tf.app.flags.DEFINE_integer("n_d_agents", 0, "Number of always defecting agents in tournament.")

# game payoffs
tf.app.flags.DEFINE_integer("reward", 3, "Number of games to train.")
tf.app.flags.DEFINE_integer("temptation", 5, "Number of games to train.")
tf.app.flags.DEFINE_integer("sucker", 0, "Number of games to train.")
tf.app.flags.DEFINE_integer("punishment", 1, "Number of games to train.")

tf.app.flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("state_size", 10, "Size of each model layer.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("model_output", "",
                           "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_output", "", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("model_path", "", "Path to trained model.weights")

FLAGS = tf.app.flags.FLAGS



def main(_):
  # Set up some parameters.
  config = FLAGS
  model_name = config.name + \
               'lr' + str(config.learning_rate) + '_' + \
               'lr_decay' + str(config.lr_decay) + '_' + \
               'n_episodes' + str(config.n_episodes) + '_' + \
               'n_batches' + str(config.n_batches) + '_' + \
               'discount' + str(config.discount) + '_' + \
               'e' + str(config.e)

  config.model_output_path = FLAGS.model_output or os.path.join('train/', model_name + '/')
  config.model_output = FLAGS.model_output or os.path.join(FLAGS.model_output_path, 'model.ckpt')
  config.log_output = FLAGS.log_output or os.path.join('log/', model_name)
  config.debug = True # debug mode
  config.n_agents = config.n_q_agents + config.n_titdat_agents + config.n_c_agents + config.n_d_agents

  # set up logger
  dirpath = os.path.dirname(config.log_output)
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)
  if not os.path.exists(config.model_output_path):
    os.makedirs(config.model_output_path)
  handler = logging.FileHandler(config.log_output)
  handler.setLevel(logging.DEBUG)
  handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
  logging.getLogger().addHandler(handler)

  logger = logging.getLogger("209_project")
  logger.setLevel(logging.DEBUG)
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

  logger.info('Config:')
  pp = pprint.PrettyPrinter(indent=1)
  logger.info(pp.pformat(config.__flags))
  logger.info('\n\n\n')

  if config.n_agents <= 0:
    logger.info('number of agents must be greater than 1')
    exit(1)

  # run model
  train.run(config)

if __name__ == "__main__":
  tf.app.run()

