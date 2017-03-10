import numpy as np


def save_scores(filepath, scores):
  scores = np.array(scores)
  n_agents = scores.shape[0]
  scores = scores.transpose([1, 0])

  with open(filepath, 'w') as file:
    # header
    file.write('i,')
    file.write(','.join(['score' + str(i) for i in range(n_agents)]))
    file.write('\n')

    # value
    for i, _scores in enumerate(scores):
      file.write(str(i) + ',')
      file.write(','.join(str(x) for x in _scores) + '\n')

  print('saved csv to: \n' + filepath)


def save_actions(filepath, actions):
  actions = np.array(actions)
  n_agents = actions.shape[0]
  actions = actions.transpose([1, 0])

  with open(filepath, 'w') as file:
    # header
    file.write('i,')
    file.write(','.join(['action' + str(i) for i in range(n_agents)]))
    file.write('\n')

    # value
    for i, _actions in enumerate(actions):
      file.write(str(i) + ',')
      file.write(','.join(str(x) for x in _actions) + '\n')

  print('saved csv to: \n' + filepath)