

class Environment(object):

  """
  https://en.wikipedia.org/wiki/Prisoner%27s_dilemma

  Canonical PD payoff matrix
              Cooperate | Defect
  Cooperate	  R, R      | S, T
  Defect	    T, S      | P, P

  R: reward
  P: punishment
  T: temptation
  S: sucker

  and to be a prisoner's dilemma game in the strong sense, the following condition must hold for the payoffs:
  T > R > P > S
  The payoff relationship R > P implies that mutual cooperation is superior to mutual defection, while the payoff relationships T > R and P > S imply that defection is the dominant strategy for both agents.

  Donation game (where cooperation is incentivized):
  Note that 2R>T+S (i.e. 2(b-c)>b-c) which qualifies the donation game to be an iterated game (see next section).
  """

  def __init__(self, config):
    self.config = config
    self.episode = 0

  def step(self, a1, a2):
    """
    action:
    0 = cooperate
    1 = defect
    """
    episode = self.episode
    self.episode += 1

    if a1 == 0 and a2 == 0:
      r1, r2 = self.config.reward, self.config.reward
    elif a1 == 0 and a2 == 1:
      r1, r2 = self.config.sucker, self.config.temptation
    elif a1 == 1 and a2 == 0:
      r1, r2 = self.config.temptation, self.config.sucker
    elif a1 == 1 and a2 == 1:
      r1, r2 = self.config.punishment, self.config.punishment

    return episode, r1, r2