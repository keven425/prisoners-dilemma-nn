import numpy as np
import time
import tensorflow as tf
import os
def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    #print('You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32,partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        #print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
def linear(in_,out_dim,name,activation_fn=None,reuse=False,init=orthogonal_initializer()):
    in_dim = in_.get_shape()[1]
    with tf.variable_scope(name,reuse=reuse):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32,initializer=init)
        b = tf.get_variable('b',[out_dim],tf.float32)
        out = tf.matmul(in_,W)+b
        if activation_fn != None:
            out = activation_fn(out)
    return out
def basic_encoder(in_,hid_dim,out_dim,scope='encoder',reuse=False,layers=1):
    with tf.variable_scope(scope,reuse=reuse):
        hid = in_
        for i in range(layers):
            hid = linear(hid,hid_dim,'hid'+str(i),tf.nn.relu)
        last_hid = linear(hid,out_dim,'hid'+str(layers))
    return last_hid
class DQN:
    def __init__(self,env,load_model=False):
        self.n_actions = env.n_actions
        self.s_dim = env.s_dim
        self.env = env
        self.replay_dim = int(1e6)
        self.burn_in = int(1e3)
        self.epsilon_end = .1
        self.anneal_frac = .2
        self.gamma = .99
        self.lr = 4e-4
        self.mb_dim = 300
        self.hid_dim = 1000
        self.tau =  1e-3
        self.layers = 3
        self.train_steps = int(1e7)
        self.refresh = int(1e4) 
        self.build_graph()
        self.setup_session()
        self.saver = tf.train.Saver()
        self.path = './network_ckpts'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if load_model:
            print('loading...')
            ckpt = tf.train.get_checkpoint_state(self.path)
            print(ckpt)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        self.sess.run(tf.global_variables_initializer())
    def build_graph(self):
        self.s = tf.placeholder(tf.float32,shape=(None,self.s_dim))
        self.a = tf.placeholder(tf.int64,shape=(None,))
        self.r = tf.placeholder(tf.float32,shape=(None,))
        self.nt = tf.placeholder(tf.float32,shape=(None,))
        self.sPrime = tf.placeholder(tf.float32,shape=(None,self.s_dim))
        self.q = basic_encoder(self.s,self.hid_dim,self.n_actions,layers = self.layers)
        selected_q = tf.reduce_sum(self.q*tf.one_hot(self.a,self.n_actions),-1)
        '''loss'''
        a_prime = tf.one_hot(tf.argmax(basic_encoder(self.sPrime,self.hid_dim,self.n_actions,reuse=True,layers=self.layers),1),self.n_actions)
        q_prime = tf.stop_gradient(tf.reduce_sum(basic_encoder(self.sPrime,self.hid_dim,self.n_actions,scope='target',layers=self.layers)*a_prime,-1))
        #q_prime = tf.stop_gradient(tf.reduce_max(basic_encoder(self.sPrime,self.hid_dim,self.n_actions,scope='target',layers=self.layers),-1))
        self.loss = tf.reduce_mean(tf.square(selected_q - (self.r+ self.gamma*self.nt*q_prime)))
        tf.summary.scalar('net loss',self.loss)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    def update_target_network(self,network_vars):
        total_vars = len(network_vars)
        ops = []
        for idx,var in enumerate(network_vars[0:total_vars//2]):
            ops.append(network_vars[idx+total_vars//2].assign((var.value()*self.tau)+((1-self.tau)*network_vars[idx+total_vars//2].value())))
        return ops
    def setup_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
    def eval(self,steps=10000,timeout = 500):
        self.env.x = np.pi/6
        self.env.v = 0.0
        s = np.asarray([self.env.x,self.env.v])
        cum_reward = 0.0
        time = 0
        for i in range(steps):
            a = np.argmax(self.sess.run(self.q,feed_dict={self.s:[s]})[0])
            sPrime,r,term = self.env.step(a)
            time+=1
            cum_reward += float(term)
            if term or time >= timeout:
                s = self.env.reset()
                time = 0
            else:
                s = sPrime
        return cum_reward/steps
    def train(self,timeout = 1000):
        update_target = self.update_target_network(tf.trainable_variables())
        cur_time = time.clock()
        avg_loss = 0
        avg_reward = 0
        rho = 1e-4
        S = np.zeros((self.replay_dim,self.s_dim))
        A = np.zeros((self.replay_dim),dtype=np.int64)
        R = np.zeros((self.replay_dim,))
        NT = np.zeros((self.replay_dim,))
        SPrime = np.zeros((self.replay_dim,self.s_dim))

        replay_ind = 0
        epsilon = np.linspace(1,self.epsilon_end,int(self.train_steps*self.anneal_frac))
        S[replay_ind] = self.env.reset()
        episode_steps = 0
        for i in range(self.train_steps):
            if np.random.rand() < epsilon[min(i,len(epsilon)-1)]:
                A[replay_ind] = np.random.randint(self.n_actions)
            else:
                A[replay_ind] = np.argmax(self.sess.run(self.q,feed_dict={self.s:[S[replay_ind]]})[0])
            SPrime[replay_ind],R[replay_ind],term = self.env.step(A[replay_ind])
            avg_reward = (1-rho)*avg_reward + (rho)*float(term)
            if i < self.train_steps-1:
                if term or episode_steps > timeout:
                    S[(replay_ind+1) % self.replay_dim] = self.env.reset()
                    NT[replay_ind] = 0
                    episode_steps = 0
                else:
                    S[(replay_ind+1) % self.replay_dim] = SPrime[replay_ind].copy()
                    NT[replay_ind] = 1
            replay_ind = (replay_ind + 1) % self.replay_dim
            episode_steps+=1
            if i > self.burn_in:
                samples = np.random.randint(min(i,self.replay_dim),size=self.mb_dim)
                _,loss = self.sess.run([self.train_step,self.loss],feed_dict={self.s:S[samples],self.a:A[samples],self.r:R[samples],self.sPrime:SPrime[samples],self.nt:NT[samples]})
                self.sess.run(update_target)
                avg_loss = (1-rho)*avg_loss + (rho)*loss
                if i % self.refresh == 0:
                    print(self.eval())
                    print('iter: ',i,' loss: ',avg_loss,' cur epsilon: ',epsilon[min(i,len(epsilon)-1)],' avg reward per timestep: ',avg_reward,' time: ',time.clock()-cur_time)
                    cur_time = time.clock()
                    loss_hist = []
                    self.saver.save(self.sess,self.path+'/model-'+str(i)+'.cptk')


if __name__ == "__main__":
    import mountain_car
    e = mountain_car.MountainCar()
    dqn = DQN(e)
    dqn.train()
