import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 20000
import matplotlib.pyplot as plt


# pretty colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
  r, g, b = tableau20[i]
  tableau20[i] = (r / 255., g / 255., b / 255.)


def get_title(config):
  agents = config.agent_names
  if len(agents) <= 2:
    return agents[0] + ' vs. ' + agents[1]
  else:
    # e.g. 1 Q-learn, 2 Titdat, 3 Coop, 4 Defect agents
    agent_counts = [
      config.n_q_agents,
      config.n_q2_agents,
      config.n_e_agents,
      config.n_titdat_agents,
      config.n_c_agents,
      config.n_d_agents
    ]
    agent_types = ['Q-1layer', 'Q-2layer', 'Extort', 'TitDat', 'Coop', 'Defect']
    agents = []
    for i, count in enumerate(agent_counts):
      if count > 0:
        agents.append(str(count) + ' ' + agent_types[i])
    name = ', '.join(agents) + ' agents'
    return 'Tournament of ' + name

  return title

def get_subtitle(config):
  sub_title = '\n\ndiscount=' + str(config.discount) + ', ' + \
    'e=' + str(config.e) + '\n' + \
    'T=' + str(config.temptation) + ', ' + \
    'R=' + str(config.reward) + ', ' + \
    'P=' + str(config.punishment) + ', ' + \
    'S=' + str(config.sucker)
  return sub_title


def moving_average(a, n=10):
  if n <= 0:
    # if length of a too small, return original array
    return a
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret / n


def markov_matrix(csv_filepath, config=None):

  actions_pairs = np.load(csv_filepath)['arr_0']
  # get an agent's action
  # _actions = np.concatenate(actions[0], axis=0)
  _actions = actions_pairs[0].transpose()
  actions1 = [item for sublist in _actions[0] for item in sublist]
  actions2 = [item for sublist in _actions[1] for item in sublist]
  _as, _is = get_markov_count(actions1, actions2)

  plt.figure(figsize=(12, 9))
  p1, = plt.plot(_is[0][0], _as[0][0], lw=1.0, color=tableau20[0], alpha=0.3)
  p2, = plt.plot(_is[0][1], _as[0][1], lw=1.0, color=tableau20[2], alpha=0.3)
  p3, = plt.plot(_is[1][0], _as[1][0], lw=1.0, color=tableau20[4], alpha=0.3)
  p4, = plt.plot(_is[1][1], _as[1][1], lw=1.0, color=tableau20[6], alpha=0.3)
  mv_avg1 = moving_average(_as[0][0], n=int(len(_as[0][0]) / 20))
  mv_avg2 = moving_average(_as[0][1], n=int(len(_as[0][1]) / 20))
  mv_avg3 = moving_average(_as[1][0], n=int(len(_as[1][0]) / 20))
  mv_avg4 = moving_average(_as[1][1], n=int(len(_as[1][1]) / 20))

  pavg1, = plt.plot(_is[0][0], mv_avg1, lw=3.5, color=tableau20[1], alpha=0.85)
  pavg2, = plt.plot(_is[0][1], mv_avg2, lw=3.5, color=tableau20[3], alpha=0.85)
  pavg3, = plt.plot(_is[1][0], mv_avg3, lw=3.5, color=tableau20[5], alpha=0.85)
  pavg4, = plt.plot(_is[1][1], mv_avg4, lw=3.5, color=tableau20[7], alpha=0.85)
  # _title = plt.suptitle(title, fontsize=24)

  ax = plt.subplot(111)
  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.set_xlabel('episode', fontsize=18)
  ax.set_ylabel('% defect', fontsize=18)
  ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=0.3)
  # ax.set_title(sub_title, fontsize=18, y=0.3)
  ttl = ax.title
  ttl.set_position([.5, 1.05])

  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  plt.tick_params(axis="both", which="both", bottom="off", top="off",
                  labelbottom="on", left="off", right="off", labelleft="on")

  plt.legend([p1, p2, p3, p4], ['CC', 'CD', 'DC', 'DD'], fontsize=18, loc='lower left') \
    .get_frame().set_linewidth(0.0)
  # plt.tight_layout()

  dirname = os.path.basename(os.path.dirname(csv_filepath))
  path = os.path.join('plots', dirname, 'markov.png')
  plt.savefig(path)
  plt.close()
  print('saved fig to: \n' + path)


def markov_matrix_prob(csv_filepath, config=None):

  actions_pairs = np.load(csv_filepath)['arr_0']
  assert config.n_agents == actions_pairs.shape[0]

  dirname = os.path.basename(os.path.dirname(csv_filepath))
  filepath = os.path.join('plots', dirname, 'markov.txt')
  with open(filepath, 'w') as fp:
    for i in range(len(actions_pairs)):
      _actions = actions_pairs[i].transpose()
      actions1 = [item for sublist in _actions[0] for item in sublist]
      actions2 = [item for sublist in _actions[1] for item in sublist]
      out_str = get_markov_out_str(actions1, actions2)

      fp.write('Agent' + str(i) + ': ' + config.agent_names[i] + '\n')
      fp.write(out_str)
      fp.write('\n\n')

  print('saved markov matrix to: \n' + filepath)


def get_markov_out_str(actions1, actions2):
  _as, _is = get_markov_count(actions1, actions2)
  p_cc = moving_average(_as[0][0], n=int(len(_as[0][0]) / 20))
  p_cd = moving_average(_as[0][1], n=int(len(_as[0][1]) / 20))
  p_dc = moving_average(_as[1][0], n=int(len(_as[1][0]) / 20))
  p_dd = moving_average(_as[1][1], n=int(len(_as[1][1]) / 20))
  p_cc = len(p_cc) and "{0:.3f}".format(p_cc[-1]) or 'n/a'
  p_cd = len(p_cd) and "{0:.3f}".format(p_cd[-1]) or 'n/a'
  p_dc = len(p_dc) and "{0:.3f}".format(p_dc[-1]) or 'n/a'
  p_dd = len(p_dd) and "{0:.3f}".format(p_dd[-1]) or 'n/a'
  out_str = 'P(CC),P(CD),P(DC),P(DD)\n' + \
            p_cc + ',' + p_cd + ',' + p_dc + ',' + p_dd
  return out_str


def get_markov_count(actions1, actions2):
  # indices.
  # e.g._is[0][1] corresponds to previously self:cooperate, opponent:defect
  _is = [
    [[], []],
    [[], []]
  ]
  # actions
  _as = [
    [[], []],
    [[], []]
  ]
  _zip = zip(actions1, actions2)
  _zip = _zip[1:]
  for i, (action1, action2) in enumerate(_zip):
    prev_a1 = _zip[i - 1][0]
    prev_a2 = _zip[i - 1][1]
    a = action1
    _is[prev_a1][prev_a2].append(i)
    _as[prev_a1][prev_a2].append(a)
  return _as, _is


def scores(npz_filepath, config):
  # its = []
  # scores = [[] for _ in range(config.n_agents)]
  title = get_title(config)
  title = 'Scores: ' + title
  sub_title = get_subtitle(config)
  # create plots/<run> if not exist
  dirname = os.path.basename(os.path.dirname(npz_filepath))
  dirpath = os.path.join('plots', dirname)
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)
  filepath = os.path.join(dirpath, 'scores.png')

  # with open(npz_filepath, 'r') as f:
  #   next(f)
  #   for line in f:
  #     splits = line.strip().split(',')
  #     its.append(int(splits[0]))
  #     for i, score in enumerate(splits[1:]):
  #       scores[i].append(float(score))
  #
  #   n_agents = len(scores)

  scores = np.load(npz_filepath)['arr_0']
  n_agents = scores.shape[0]
  assert n_agents == config.n_agents

  its = range(scores.shape[1])
  legends = [("Score: " + agent) for agent in config.agent_names]
  plot(filepath, its, scores, title, sub_title, legends, 'round', 'score', config)

def actions(npz_filepath, config):
  title = get_title(config)
  title  = 'Actions: ' + title
  sub_title = get_subtitle(config)
  dirname = os.path.basename(os.path.dirname(npz_filepath))
  filepath = os.path.join('plots', dirname, 'actions.png')

  actions = np.load(npz_filepath)['arr_0']
  its = range(actions.shape[1])
  # its, actions1, actions2 = read_actions(csv_filepath, config)
  legends = [("% Defect: " + agent) for agent in config.agent_names]
  plot(filepath, its, actions, title, sub_title, legends, 'episode', 'defect = 1, cooperate = 0', config)


# def read_actions(csv_filepath, config):
#   its = []
#   actions = [[] for _ in range(config.n_agents)]
#   with open(csv_filepath, 'r') as f:
#     next(f)
#     for line in f:
#       splits = line.strip().split(',')
#       its.append(int(splits[0]))
#       for i, action in enumerate(splits[1:]):
#         actions[i].append(float(action))
#
#     n_agents = len(actions)
#     assert n_agents == config.n_agents
#
#   return its, actions


def plot(filepath, x, y_s, title, sub_title, legends, x_label, y_label, config):

  plt.figure(figsize=(12, 9))
  ps = []
  for i, y in enumerate(y_s):
    p, = plt.plot(x, y_s[i], lw=1.0, color=tableau20[i], alpha=0.3)
    ps.append(p)
  window = int(len(y_s[0]) / 20)
  p_avgs = []
  for i, y in enumerate(y_s):
    mv_avg = moving_average(y_s[i], n=window)
    p, = plt.plot(x, mv_avg, lw=2.5, color=tableau20[i], alpha=0.85)
    p_avgs.append(p)
  _title = plt.suptitle(title, fontsize=24)
  # _title.set_position([.5, .2])
  # _sub_title = plt.title(sub_title, fontsize=18, y=0.3)
  # _sub_title.set_position([.5, 0.])

  ax = plt.subplot(111)
  ax.spines["top"].set_visible(False)
  ax.spines["bottom"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_visible(False)
  ax.set_xlabel(x_label, fontsize=18)
  ax.set_ylabel(y_label, fontsize=18)
  ax.yaxis.grid(b=True, which='major', color='black', linestyle='--', alpha=0.3)
  ax.set_title(sub_title, fontsize=18, y=0.3)
  ttl = ax.title
  ttl.set_position([.5, 1.05])

  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  plt.tick_params(axis="both", which="both", bottom="off", top="off",
                  labelbottom="on", left="off", right="off", labelleft="on")

  plt.legend(ps, legends, fontsize=18, loc='lower left')\
    .get_frame().set_linewidth(0.0)
  plt.tight_layout()

  plt.savefig(filepath)
  plt.close()
  print('saved fig to: \n' + filepath)


if __name__ == "__main__":
  # tiny config class
  class Config:
    def __init__(self, **kwargs):
      self.__dict__.update(kwargs)

  config = Config(n_agents=2, agent_names=['Q-1layer', 'TitDat'])
  markov_matrix_prob('train/titdat_qa1_ta1_state10_lr0.01_lr_decay0.9995_n_episodes10_n_batches20000_discount0.95_e0.2_adapt0.9999_r3_t5_s0_p1/actions_pair.npz',
                     config)


def agent_log(log, csv_filepath):
  dirname = os.path.basename(os.path.dirname(csv_filepath))
  filepath = os.path.join('plots', dirname, 'agent_log.txt')
  with open(filepath, 'w') as fp:
    fp.write(log)
  print('saved log to ' + filepath)