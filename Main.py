# default carla scripts
import cv2

import __generate_traffic as traffic
from CarlaEnvironments import *

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

with CarlaEnvironment(port=4242, delta_seconds=1/30) as env:
    env.change_map("town03")
    vehicle = Vehicle(env, "model3", True, 0)
    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    camera.actor.listen(lambda image: save_image_memory(image))

    for i in range(30):
        Vehicle(env, None, True, None)

    for i in range(30):
        Pedestrian(env, None)

    while True:
        begin = time.time()
        show_image_cv2()
        env.step()
        print("frame time: {:.2f}ms, simulation time: {:.1f} seconds".format((time.time()-begin)*1000, env.simulated_time))

