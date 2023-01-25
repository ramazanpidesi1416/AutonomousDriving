# default carla scripts
import cv2

import __generate_traffic as traffic
from CarlaEnvironments import *

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

with CarlaEnvironment(port=4242, delta_seconds=1/30) as env:
    env.change_map("town03")
    vehicle = Vehicle(env, None, True, 0)
    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    camera.actor.listen(lambda image: save_image_memory(image))

    for i in range(40):
        Vehicle(env, None, True)

    for i in range(40):
        Pedestrian(env)

    while True:
        env.step()
        show_image_cv2()


