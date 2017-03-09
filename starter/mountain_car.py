import numpy as np
def normalize(s):
    x = s[0]
    v = s[1]
    x = (x-MountainCar.min_x)/(MountainCar.max_x-MountainCar.min_x)
    v = (v-MountainCar.min_v)/(MountainCar.max_v-MountainCar.min_v)
    return np.asarray([x,v])
def unnormalize(s):
    x = s[0]
    v = s[1]
    x = x*(MountainCar.max_x-MountainCar.min_x)+MountainCar.min_x
    v = v*(MountainCar.max_v-MountainCar.min_v)+MountainCar.min_v
    return np.asarray([x,v])
class MountainCar(object):
    min_x = -1.2
    max_x = .5
    min_v = -.07
    max_v = .07
    def __init__(self):
        self.n_actions = 2
        self.s_dim = 2
        self.reset()
    def reset(self):
        self.x = np.random.rand()*(self.max_x-self.min_x)+self.min_x
        self.v = np.random.rand()*(self.max_v-self.min_v)+self.min_v
        return normalize(np.asarray([self.x,self.v]))
    def step(self,action):
        self.v = self.v + .001*(2*action-1) - .0025*np.cos(3*self.x)
        self.v = max(self.min_v,min(self.v,self.max_v))
        self.x = self.x + self.v
        self.x = max(self.min_x,min(self.x,self.max_x))
        reward = -1.0
        term = False
        if self.x == self.max_x:
            reward = 0.0
            term = True
        return normalize(np.asarray([self.x,self.v])),reward,term 
if __name__ == "__main__":
    import dqn
    def policy(s):
        x = s[0]
        v = s[1]
        if v < 0 and x != -1.2:
            action = 0
        else:
            action = 1
        return action
    import matplotlib.pyplot as plt
    plt.ion()
    env = MountainCar()
    model = dqn.DQN(env,True)
    def network_policy(s):
        q_values = model.sess.run(model.q,feed_dict={model.s:[s]})[0]
        return np.argmax(q_values),q_values
    s = env.reset()
    cum_r = 0.0
    steps = int(1e4)
    episode_length = 0
    for i in range(steps):
        #a = np.random.randint(3)
        a,q_values = network_policy(s)
        s,r,term = env.step(a)
        print(s)
        x,v = unnormalize(s)[:]
        cum_r+=r
        plt.clf()
        plt.subplot(121)
        s_pos = np.linspace(env.min_x,env.max_x)
        plt.plot(s_pos,np.sin(3*s_pos)/3)
        plt.scatter(x,np.sin(3*x)/3)
        plt.subplot(122)
        plt.bar([-1,1],q_values)
        plt.pause(.001)
        episode_length+=1
        if term:
            print('win')
            s = env.reset()
        if episode_length > 100:
            s = env.reset()
            episode_length = 0
            print('out of time')
    print(cum_r/steps)
