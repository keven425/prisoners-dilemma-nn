import os
import numpy as np
import matplotlib.pyplot as plt

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
  title = config.game_name
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
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret / n


def markov_matrix(csv_filepath, config=None):

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

  its, actions1, actions2 = read_actions(csv_filepath)
  _zip = zip(actions1, actions2)
  _zip = _zip[1:]
  for i, (action1, action2) in enumerate(_zip):
    prev_a1 = _zip[i - 1][0]
    prev_a2 = _zip[i - 1][1]
    a = action1
    _is[prev_a1][prev_a2].append(i)
    _as[prev_a1][prev_a2].append(a)

  plt.figure(figsize=(12, 9))
  p1, = plt.plot(_is[0][0], _as[0][0], lw=1.0, color=tableau20[0], alpha=0.3)
  p2, = plt.plot(_is[0][1], _as[0][1], lw=1.0, color=tableau20[2], alpha=0.3)
  p3, = plt.plot(_is[1][0], _as[1][0], lw=1.0, color=tableau20[4], alpha=0.3)
  p4, = plt.plot(_is[1][1], _as[1][1], lw=1.0, color=tableau20[6], alpha=0.3)
  window = int(len(its) / 20)
  mv_avg1 = moving_average(_as[0][0], n=window)
  mv_avg2 = moving_average(_as[0][1], n=window)
  mv_avg3 = moving_average(_as[1][0], n=window)
  mv_avg4 = moving_average(_as[1][1], n=window)
  pavg1, = plt.plot(_is[0][0], mv_avg1, lw=2.5, color=tableau20[1], alpha=0.85)
  pavg2, = plt.plot(_is[0][1], mv_avg2, lw=2.5, color=tableau20[3], alpha=0.85)
  pavg3, = plt.plot(_is[1][0], mv_avg3, lw=2.5, color=tableau20[5], alpha=0.85)
  pavg4, = plt.plot(_is[1][1], mv_avg4, lw=2.5, color=tableau20[7], alpha=0.85)
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



def scores(csv_filepath, config):
  its = []
  scores1 = []
  scores2 = []
  title = get_title(config)
  title = 'Scores: ' + title
  sub_title = get_subtitle(config)
  # create plots/<run> if not exist
  dirname = os.path.basename(os.path.dirname(csv_filepath))
  dirpath = os.path.join('plots', dirname)
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)
  filepath = os.path.join(dirpath, 'scores.png')

  with open(csv_filepath, 'r') as f:
    next(f)
    for line in f:
      it, score1, score2 = line.strip().split(',')
      its.append(int(it))
      scores1.append(float(score1))
      scores2.append(float(score2))

  plot(filepath, its, (scores1, scores2), title, sub_title, ["Score: Agent 1", "Score: Agent 2"], 'round', 'score', config)

def actions(csv_filepath, config):
  title = get_title(config)
  title  = 'Actions: ' + title
  sub_title = get_subtitle(config)
  dirname = os.path.basename(os.path.dirname(csv_filepath))
  filepath = os.path.join('plots', dirname, 'actions.png')

  its, actions1, actions2 = read_actions(csv_filepath)
  plot(filepath, its, (actions1, actions2), title, sub_title, ["% Defect: Agent 1", "% Defect: Agent 2"], 'episode', 'defect = 1, cooperate = 0', config)


def read_actions(csv_filepath):
  its = []
  actions1 = []
  actions2 = []
  with open(csv_filepath, 'r') as f:
    next(f)
    for line in f:
      it, action1, action2 = line.strip().split(',')
      its.append(int(it))
      actions1.append(int(action1))
      actions2.append(int(action2))
  return its, actions1, actions2


def plot(filepath, x, y_s, title, sub_title, legends, x_label, y_label, config):

  plt.figure(figsize=(12, 9))
  p1, = plt.plot(x, y_s[0], lw=1.0, color=tableau20[0], alpha=0.3)
  p2, = plt.plot(x, y_s[1], lw=1.0, color=tableau20[7], alpha=0.3)
  window = int(len(y_s[0]) / 20)
  mv_avg1 = moving_average(y_s[0], n=window)
  mv_avg2 = moving_average(y_s[1], n=window)
  p3, = plt.plot(x, mv_avg1, lw=2.5, color=tableau20[0], alpha=0.85)
  p4, = plt.plot(x, mv_avg2, lw=2.5, color=tableau20[7], alpha=0.85)
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

  plt.legend([p1, p2], legends, fontsize=18, loc='lower left')\
    .get_frame().set_linewidth(0.0)
  plt.tight_layout()

  plt.savefig(filepath)
  plt.close()
  print('saved fig to: \n' + filepath)


if __name__ == "__main__":
  markov_matrix('train/lr0.05_lr_decay0.9995_n_episodes10_n_batches99999_discount0.95_e0.2/actions.csv')