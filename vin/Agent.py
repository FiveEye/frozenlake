from collections import deque
import random
import numpy as np
import keras as ks
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
import keras.backend as K

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORATION_STEPS = 10000
NUM_REPLAY_MEMORY = 40000
INITIAL_REPLAY_SIZE = 1000
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 11
TRAIN_INTERVAL = 3

def Q_VIN(sz, ch_i, k, ch_h, ch_q, ch_a):
    map_in = Input(shape=(sz,sz,ch_i))
    s = Input(shape=(1,), dtype='int32')
    #print(s)
    h = Conv2D(filters=ch_h, 
               kernel_size=(3,3), 
               padding='same', 
               activation='relu')(map_in)

    r = Conv2D(filters=1, 
               kernel_size=(3,3), 
               padding='same',
               use_bias=False,
               activation=None,
               )(h)

    conv3 = Conv2D(filters=ch_q, 
                   kernel_size=(3,3), 
                   padding='same',
                   use_bias=False)

    conv3b = Conv2D(filters=ch_q, 
                    kernel_size=(3,3), 
                    padding='same',
                    use_bias=False)

    #v = Lambda(lambda x:  x[:,:,:,1:], output_shape=(sz,sz,1))(map_in)
    #print(v)
    #rv = concatenate([r, v], axis=3)
    #q = conv3b(rv)
    q = conv3(r,)
    for _ in range(k):
        v = Lambda(lambda x: K.max(x, axis=3, keepdims=True), output_shape=(sz,sz,1))(q)
        rv = concatenate([r, v], axis=3)
        q = conv3b(rv)
    
    #print(q)
    q = Reshape(target_shape=(sz * sz, ch_q))(q)
    #print(q)
    
    def attention(x):
        N = K.shape(x)[0]
        q_out = K.map_fn(lambda i: K.gather(x[i], s[i,0]), K.arange(0,N), dtype='float32')
        return q_out

    q_out = Lambda(attention, output_shape=(ch_q,))(q)

    out = Dense(units=ch_a, input_shape=(ch_q,), activation='linear', use_bias=False)(q_out)
    
    v = Reshape(target_shape=(sz * sz, 1))(v)
    v_out = Lambda(attention, output_shape=(1,))(v)
    return Model(inputs=[map_in,s], outputs=out), [map_in, s], out, Model(inputs=[map_in,s], outputs=v_out)

def train_Model(ms, out):
    map_in, s = ms
    a = Input(shape=(1,), dtype='int32')
    def attention(x):
        #x = K.permute_dimensions(x, (1,0,2))
        N = K.shape(x)[0]
        a_out = K.map_fn(lambda i: K.gather(x[i], a[i,0]), K.arange(0,N), dtype='float32')
        return a_out
    a_out = Lambda(attention, output_shape=(1,))(out)
    #print('a_out', a_out)
    return Model(inputs=[map_in, s, a], outputs=a_out)


class Agent:
    def __init__(self, env):
        self.num_obs = env.observation_space.n
        self.map_sz = np.int(np.sqrt(self.num_obs))

        self.init_map()

        self.num_actions = env.action_space.n
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS


        self.q_nn, self.q_s, self.q_out, self.v_model     = Q_VIN(self.map_sz, 2, self.map_sz+6, 150, 10, self.num_actions)
        self.target_nn, self.target_s, self.target_out, _ = Q_VIN(self.map_sz, 2, self.map_sz+6, 150, 10, self.num_actions)

        self.train_model = train_Model(self.q_s, self.q_out)

        self.t = 0
        self.total_reward = 0
        self.episode = 0

        self.replay_memory = deque()
        self.good_memory = []
        self.bad_memory = []

        self.q_nn.compile(loss='mean_squared_error',
                          optimizer='RMSprop',
                          metrics=['accuracy'])

        self.target_nn.compile(loss='mean_squared_error',
                               optimizer='RMSprop',
                               metrics=['accuracy'])
        
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer='RMSprop',
                                 metrics=['accuracy'])

        self.v_model.compile(loss='mean_squared_error',
                                 optimizer='RMSprop',
                                 metrics=['accuracy'])

    def init_map(self):
        self.map_info = np.repeat(np.array([[0.0,0.0]], dtype='float32'), self.num_obs, axis=0)
        self.map_info = np.reshape(self.map_info, (self.map_sz, self.map_sz, 2))
    
    def print_map_info(self):
        str = 'map_info\n'
        for i in range(0, self.map_sz):
            for j in range(0, self.map_sz):

                if self.map_info[i,j,0] > 0.0:
                    str += 'H'
                elif self.map_info[i,j,1] > 0.0:
                    str += 'G'
                else:
                    str += 'F'
            str += '\n'
        print(str)

    def init(self, observation):
        x = observation // self.map_sz
        y = observation % self.map_sz
        self.map_info[x,y,0] = 0
        self.map_info[x,y,1] = 0

    def updateTarget(self):
        self.target_nn.set_weights(self.q_nn.get_weights())

    def get_action(self, state):
        action = random.randrange(self.num_actions)
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            pred = self.q_nn.predict([np.array([self.map_info]), np.array([state])])
            print(self.epsilon)
            print(pred)
            #for i in range(self.num_actions):
            #    print(self.train_model.predict([np.array([self.map_info]), np.array([state]), np.array([i])]))
            action = np.argmax(pred)

        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step
        return action

    def run(self, state, action, reward, terminal, observation):
        x = observation // self.map_sz
        y = observation % self.map_sz
        if terminal == False:
            reward = 0.0
        else:
            if reward > 0:
                reward = 1.0
                self.map_info[x,y,1] = 1.0
            else:
                reward = -1.0
                self.map_info[x,y,0] = 1.0

        self.replay_memory.append((state, action, reward, observation, terminal))
        if reward > 0.0:
            self.good_memory.append((state, action, reward, observation, terminal))
        elif terminal:
            self.bad_memory.append((state, action, reward, observation, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.t >= INITIAL_REPLAY_SIZE:
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.updateTarget()
        #if terminal:
        #    print(reward)
        #    print(self.map_info)
        self.t += 1



    def train_network(self):
        map_batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        if len(self.good_memory) < BATCH_SIZE:
            minibatch = self.good_memory
        else:
            minibatch = random.sample(self.good_memory, BATCH_SIZE)
        for data in minibatch:
            map_batch.append(self.map_info)
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        if len(self.bad_memory) < BATCH_SIZE:
            minibatch = self.bad_memory
        else:
            minibatch = random.sample(self.bad_memory, BATCH_SIZE)
        for data in minibatch:
            map_batch.append(self.map_info)
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            map_batch.append(self.map_info)
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])


        map_batch = np.array(map_batch)
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)

        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_nn.predict([map_batch, next_state_batch])
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)


        #print(target_q_values_batch.shape, y_batch.shape)
        self.v_model.fit([map_batch, next_state_batch], y_batch)
        self.train_model.fit([map_batch, state_batch, action_batch], y_batch)
        

        #target_q_values_batch = self.target_nn.predict([np.array(map_batch), np.array(next_state_batch)])
        #y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)


        #self.q_nn.fit([np.array(map_batch), np.array(state_batch)], np.array(y_batch))
