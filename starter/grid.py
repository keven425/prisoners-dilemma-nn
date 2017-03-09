import numpy as np
import matplotlib.pyplot as plt
class Grid(object):
    def __init__(self,verbose=False):
        if verbose:
            plt.ion()
            self.grid_image = np.zeros((25,))
        self.verbose=verbose
        self.reset()
    def render(self):
        self.grid_image[:] = 0
        self.grid_image[self.s] = 1
        plt.clf()
        plt.imshow(self.grid_image.reshape([5,5]))
        plt.pause(.001)
    def reset(self):
        self.s = 4*5+0
        return self.s
    def step(self,action):
        row = int(self.s/5)
        col = self.s % 5
        if action == 0:
            col = min(col+1,4)
        elif action == 1:
            row = max(row-1,0)
        elif action == 2:
            col = max(col-1,0)
        elif action == 3:
            row = min(row+1,4)
        self.s = row*5+col
        if row == 4 and col ==4:
            reward = 1.0
            term = True
        elif col > 0 and col < 4 and row > 2:
            reward = -1.0
            term = True
        else:
            reward = 0
            term = False
        if self.verbose:
            self.render()
        return self.s,reward,term

