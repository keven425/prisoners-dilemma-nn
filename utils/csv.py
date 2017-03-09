
def save_scores(filepath, scores):
  scores = zip(*scores)

  with open(filepath, 'w') as file:

    file.write('i,score1,score2\n')
    for i, _scores in enumerate(scores):
      file.write(str(i) + ',' + str(_scores[0]) + ',' + str(_scores[1]) + '\n')

  print('saved csv to: \n' + filepath)


def save_actions(filepath, actions):
  actions = zip(*actions)

  with open(filepath, 'w') as file:

    file.write('i,action1,action2\n')
    for i, _actions in enumerate(actions):
      file.write(str(i) + ',' + str(_actions[0]) + ',' + str(_actions[1]) + '\n')

  print('saved csv to: \n' + filepath)