import carla
import random
import time
import glob
import os
import sys
import numpy as np
import cv2

# setup of system variables for carla
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480


class CarlaEnvironment:

    def __init__(self):
        self.initialization_successful = False
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.delta_seconds = 1 / 30.0

        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        self.actor_list = []
        self.initialization_successful = True

    def __del__(self):
        if not self.initialization_successful:
            return
        for client in self.actor_list:
            client.destroy()
        print("destructor is called")

    def step(self):
        self.frame = self.world.tick()

    def get_map_spawnpoints(self):
        return self.world.get_map().get_spawn_points()


class Vehicle:
    def __init__(self, carla_environment, vehicle_type="model3"):
        spawn_point = carla_environment.get_map_spawnpoints()[0]
        bp = env.blueprint_library.filter(vehicle_type)[0]
        self.actor = env.world.spawn_actor(bp, spawn_point)
        carla_environment.actor_list.append(self.actor)

    def apply_control(self, carla_vehicle_control):
        self.actor.apply_control(carla_vehicle_control)

    def apply_control(self, throttle=0, steer=0, brake=0.0, ):
        pass

class Camera:
    pass


def process_image(image):
    image1 = np.array(image.raw_data, np.uint8)
    image2 = image1.reshape((IM_HEIGHT, IM_WIDTH, 4))
    image3 = image2[:, :, :3]
    print("image read")
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview", image3)
    cv2.waitKey(40)
    return image3/255.0

image_count = 0
def save_image(image):
    global image_count
    image.save_to_disk("camera_data/image%06d.png" % image_count)
    image_count += 1
    print(image_count)


env = CarlaEnvironment()
try:
    vehicle = Vehicle(env, "model3")
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0))

    cam_bp = env.blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = env.world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle.actor)
    env.actor_list.append(sensor)

    sensor.listen(lambda image: save_image(image))
    while True:
        env.step()
finally:
    env.__del__()   # deletes all actors created during the execution of this script.





