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

class CarlaEnvironment:

    def __init__(self, delta_seconds=1/30.0, no_rendering_mode=False, synchronous_mode=True):
        self.initialization_successful = False
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.delta_seconds = delta_seconds
        self.no_rendering_mode = no_rendering_mode
        self.synchronous_mode = synchronous_mode

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

        self.change_settings(delta_seconds=1/30.0, no_rendering_mode=False, synchronous_mode=False)
        print("scene properly cleaned")

    def step(self):
        self.frame = self.world.tick()

    def get_map_spawnpoints(self):
        return self.world.get_map().get_spawn_points()

    # None will not change the setting
    def change_settings(self, delta_seconds=None, no_rendering_mode=None, synchronous_mode=None):
        if delta_seconds is not None:
            self.delta_seconds = delta_seconds
        if no_rendering_mode is not None:
            self.no_rendering_mode = no_rendering_mode
        if synchronous_mode is not None:
            self.synchronous_mode = synchronous_mode
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=self.no_rendering_mode,
            synchronous_mode=self.no_rendering_mode,
            fixed_delta_seconds=self.delta_seconds))


class Vehicle:
    def __init__(self, carla_environment, vehicle_type="model3"):
        spawn_point = carla_environment.get_map_spawnpoints()[0]
        bp = carla_environment.blueprint_library.filter(vehicle_type)[0]
        self.actor = carla_environment.world.spawn_actor(bp, spawn_point)
        carla_environment.actor_list.append(self.actor)

    def apply_control(self, carla_vehicle_control):
        self.actor.apply_control(carla_vehicle_control)

    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0):
        self.actor.apply_control(carla.VehicleControl(throttle=throttle,
                                                      steer=steer,
                                                      brake=brake,
                                                      hand_brake=hand_brake,
                                                      reverse=reverse,
                                                      manual_gear_shift=manual_gear_shift,
                                                      gear=gear))


class vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Camera:

    def __init__(self, carla_environment, attaching_carla_actor, image_width=640, image_height=480, fov=110, displacement=vector(2.5, 0, 0.7)):
        self.blueprint = carla_environment.blueprint_library.find('sensor.camera.rgb')
        self.blueprint.set_attribute("image_size_x", f"{image_width}")
        self.blueprint.set_attribute("image_size_y", f"{image_height}")
        self.blueprint.set_attribute("fov", f"{fov}")
        self.displacement = carla.Transform(carla.Location(displacement.x, displacement.y, displacement.z))
        self.actor = carla_environment.world.spawn_actor(self.blueprint, self.displacement, attach_to=attaching_carla_actor)
        carla_environment.actor_list.append(self.actor)




