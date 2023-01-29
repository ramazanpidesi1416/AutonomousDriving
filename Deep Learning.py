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

def env_reset(carla_environment):
    carla_environment.clear_objects()
    vehicle = Vehicle(carla_environment, "model3", False, 0)
    return vehicle

class TrainingEnvironment:

    def __init__(self, delta_seconds=1/15.0, no_rendering_mode=False, port=2000):
        self.delta_seconds = delta_seconds
        self.no_rendering_mode = no_rendering_mode
        self.port = port

        self.simulation_duration = 30  # runs are 300 seconds long (in-simulation time)

        self.carla_env = None
        self.vehicle = None
        self.reset()


    def step(self, action): # actions are only an index, fix it
        throttle = 0
        steer = 0
        brake = 0
        if action == 0:
            throttle = 0
            brake = 0
            steer = 0
        elif action == 1:
            throttle = 0.6
            brake = 0
            steer = 0
        elif action == 2:
            throttle = 0.4
            steer = 1
            brake = 0
        elif action == 3:
            throttle = 0.4
            steer = -1
            brake = 0
        elif action == 4:
            throttle = 0
            steer = 0
            brake = 1


        self.vehicle.apply_control(throttle=throttle, steer=steer, brake=brake)
        self.carla_env.step(True)
        next_state = self._get_state()

        reward = self._get_reward()
        print(reward)

        done = self.carla_env.simulated_time >= self.simulation_duration
        return next_state, reward, done, None

    def reset(self):
        self.carla_env = CarlaEnvironment(delta_seconds=self.delta_seconds, no_rendering_mode=self.no_rendering_mode, port=self.port)
        self.carla_env.change_map('town03')
        self.vehicle = Vehicle(self.carla_env, spawn_point=0)
        return self._get_state()

    def _get_state(self):
        position = self.vehicle.get_position()
        return [position.x, position.y, position.z]

    def _get_reward(self):
        current_position = self.vehicle.get_position()
        return current_position.length()

class DQNAgent:

    def __init__(self):
        self.env = TrainingEnvironment(delta_seconds=1/15, no_rendering_mode=False, port=4565)

        self.state_size = 3
        self.action_size = 5
        self.EPISODES = 1000
        self.memory = deque(maxlen=1024 * 4)

        self.gamma = 0.95  # discount rate, how important long term rewards are compared the short term rewards, 1 means identical, 0 means have no importance
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.92
        self.batch_size = 1024
        self.epoch_count = 4
        self.train_start = 1024

        # create main model
        self.model = generate_model(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decay_epsilon(self):
        if len(self.memory) >= self.train_start:
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0))

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
        self.model.fit(state, target, batch_size=self.batch_size, epochs=self.epoch_count, verbose=1)
        self.decay_epsilon()

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
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done:
                    reward = reward
                else:
                    reward = -100   # I'm not sure about this

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    print("Saving trained model as dqn.h5")
                    self.save("simple dqn.h5")
                    #return
            self.replay()


agent = DQNAgent()
while True:
    agent.run()
