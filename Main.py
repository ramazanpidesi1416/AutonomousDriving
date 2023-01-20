# default carla scripts
import cv2

import __generate_traffic as traffic
from CarlaEnvironments import *

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

env = CarlaEnvironment()
try:
    vehicle = Vehicle(env, None, True, 0)

    for i in range(30):
         Vehicle(env, None, True)

    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    # vehicle.apply_control(throttle=1.0)

    camera.actor.listen(lambda image: save_image_memory(image))

    while True:
        env.step()
        show_image_cv2()

finally:
    env.clear_objects()  # deletes all actors created during the execution of this script.
