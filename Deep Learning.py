from CarlaEnvironments import *

import keras
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.models import Model, load_model
import tensorflow as tf
from collections import deque

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate a section of gpu
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 16)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

"""
def generate_model():
    model = keras.Sequential()
    model.add(Conv2D(8, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu', input_shape=[IM_HEIGHT, IM_WIDTH, 3]))
    model.add(Conv2D(16, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
    model.add(Conv2D(16, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
    model.add(Conv2D(16, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
    model.add(Conv2D(1, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['Accuracy'])
    model.summary()
    return model
"""

def generate_model(input_shape, action_space):
    model = keras.Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dense(action_space, activation='sigmoid', input_shape=input_shape))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['Accuracy'])
    model.summary()
    return model

vehicle = None
def env_reset(carla_environment):
    global vehicle
    carla_environment.clear_objects()
    vehicle = Vehicle(carla_environment, "model3", False, 0)

class DQNAgent:

    def __init__(self):
        # by default, CartPole-v1 has max episode steps = 500
        self.env = CarlaEnvironment(delta_seconds=1/15, no_rendering_mode=False, port=4221)
        self.env.change_map("town03")
        self.vehicle = Vehicle(self.env, spawn_point=0)

        self.state_size = 2
        self.action_size = 4
        self.EPISODES = 1000
        self.memory = deque(maxlen=1024 * 4)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.999
        self.batch_size = 1024
        self.epoch_count = 4
        self.train_start = 1024
        self.visualize = True
        self.image_width = 64
        self.image_height = 64

        # create main model
        self.model = generate_model(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.train_start:
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0))

    def step(self):
        state = self.vehicle.get_position()
        self.env.step(True)
        control = self.vehicle.actor.get_control()
        next_state = self.vehicle.get_position()
        self.vehicle.update_total_distance_travelled()
        reward = self.vehicle.total_distance_travelled # TODO: define reward
        return [state.x, state.y, state.z], [control.throttle, control.steer, control.brake], reward, [next_state.x, next_state.y, next_state.z]

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state, verbose=0)
        target_next = self.model.predict(next_state, verbose=0)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, epochs=self.epoch_count, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                if self.visualize:
                    self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    print("Saving trained model as dqn.h5")
                    self.save("dqn.h5")
                    #return
            self.replay()


agent = DQNAgent()
while True:
    agent.step()
