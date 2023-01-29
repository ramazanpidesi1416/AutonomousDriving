import numpy as np

from CarlaEnvironments import *

import keras
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 16)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# hyper parameters
IM_WIDTH = 64
IM_HEIGHT = 64

batch_size = 1024 * 4

model = keras.Sequential()
model.add(Conv2D(8, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu', input_shape=[IM_HEIGHT, IM_WIDTH, 3]))
model.add(Conv2D(16, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
model.add(Conv2D(16, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
model.add(Conv2D(16, kernel_size=[3, 3], strides=[1, 1], padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='linear'))
model.compile(optimizer='Adam', loss=MeanSquaredError(), metrics=['Accuracy'])
model.summary()

model.load_weights('first_model')

# 0-> throttle 1-> steer 2-> brake
controls = np.zeros(shape=[batch_size, 3])
images = np.zeros(shape=[batch_size, IM_HEIGHT, IM_WIDTH, 3], dtype='uint8')

with CarlaEnvironment(port=4221, delta_seconds=1/60, no_rendering_mode=False) as env:
    env.change_map("town03")
    vehicle = Vehicle(env, "model3", False, 0)
    camera = Camera(env, vehicle.actor, "semantic", IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    gnss = GNSS(env, vehicle.actor)
    collusion = CollusionSensor(env, vehicle.actor)
    lane_invasion = LaneInvasionSensor(env, vehicle.actor)

    for i in range(30):
        Vehicle(env, None, True, None)

    for i in range(30):
        Pedestrian(env, None)

    for i in range(batch_size):
        env.step(True)
        camera.display_data("camera1")

        control = vehicle.actor.get_control()
        controls[i] = [control.throttle, control.steer, control.brake]

        images[i] = camera.camera_image
        action = model.predict(np.reshape(camera.camera_image, [1, IM_HEIGHT, IM_WIDTH, 3]))[0]
        print(action)
        vehicle.apply_control(throttle=float(action[0]), steer=float(action[1]), brake=max(float(action[2] - 0.07), 0))


#for i in range(10):
#    model.fit(images, controls, epochs=1)
#    model.save_weights('first_model')
