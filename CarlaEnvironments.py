import carla
import random
import time
import glob
import os
import sys
import numpy as np
import cv2
import argparse
import threading

# carla default scripts
import __generate_traffic


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
        self.deleted = False
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
        print("scene initialization is successful")


    def __del__(self):
        print("destructor is called")
        self.change_settings(delta_seconds=1 / 30.0, no_rendering_mode=False, synchronous_mode=False)
        try:
            self.clear_objects()
        except:
            pass

    def clear_objects(self):
        print("clearing the environment")
        for client in self.actor_list:
            client.destroy()
        print("clear successful")

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


class vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Vehicle:
    """spawn_point can be a vector or an index, if given an index it will be i'th default spawn point in carla map, if left as None spawn point will be picked random"""
    def __init__(self, carla_environment, vehicle_type="model3", autopilot=False, spawn_point=None):
        try:
            if type(spawn_point) == vector:
                spawn_point = carla.Transform(spawn_point.x, spawn_point.y, spawn_point.z)
            elif type(spawn_point) == int:
                spawn_point = carla_environment.get_map_spawnpoints()[spawn_point]
            elif spawn_point is None:
                spawn_point = random.choice(carla_environment.get_map_spawnpoints())
            else:
                print("spawn_point must be a int, vector or None")
                raise Exception

            if type(vehicle_type) == str:
                bp = carla_environment.blueprint_library.filter(vehicle_type)[0]
            elif vehicle_type is None:
                bp = random.choice(carla_environment.blueprint_library.filter('vehicle.*.*'))
            else:
                print("vehicle type must be a string or None")
                raise Exception

            self.actor = carla_environment.world.spawn_actor(bp, spawn_point)
        except Exception:
            print("vehicle couldn't initialized")
            return

        carla_environment.actor_list.append(self.actor)
        self.autopilot = autopilot
        self.actor.set_autopilot(self.autopilot)



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


class Pedestrian:
    def __init__(self, carla_environment, start_position=None):
        try:
            self.bp = random.choice(carla_environment.world.get_blueprint_library().filter('*walker.pedestrian*'))
            if start_position is None:
                self.transform = carla.Transform(carla_environment.world.get_random_location_from_navigation())
            else:
                self.transform = carla.Transform(carla.Location(x=start_position.x, y=start_position.y, z=start_position.z))

            self.actor = carla_environment.world.try_spawn_actor(self.bp, self.transform)
            self.controller_bp = carla_environment.world.get_blueprint_library().find('controller.ai.walker')
            self.controller = carla_environment.world.spawn_actor(self.controller_bp, self.actor.get_transform(), self.actor)
        except:
            print("Pedestrian couldn't initialized")
            return
        carla_environment.step()
        self.controller.start()
        self.controller.go_to_location(carla_environment.world.get_random_location_from_navigation())
        self.controller.set_max_speed(1 + random.random())
        carla_environment.actor_list.append(self.actor)
        carla_environment.actor_list.append(self.controller)


class Camera:

    def __init__(self, carla_environment, attaching_carla_actor, image_width=640, image_height=480, fov=110, displacement=vector(2.5, 0, 0.7)):
        self.blueprint = carla_environment.blueprint_library.find('sensor.camera.rgb')
        self.blueprint.set_attribute("image_size_x", f"{image_width}")
        self.blueprint.set_attribute("image_size_y", f"{image_height}")
        self.blueprint.set_attribute("fov", f"{fov}")
        self.displacement = carla.Transform(carla.Location(displacement.x, displacement.y, displacement.z))
        self.actor = carla_environment.world.spawn_actor(self.blueprint, self.displacement, attach_to=attaching_carla_actor)
        carla_environment.actor_list.append(self.actor)

    def get_image(self):
        return self.actor



def generate_traffic(asynch=False, car_lights_on=False, filterv="vehicle.*", filterw='walker.pedestrian.*',
                              generationv='All', generationw='2', hero=False, host='127.0.0.1', hybrid=False,
                              no_rendering=False, number_of_vehicles=30, number_of_walkers=10, port=2000, respawn=False,
                              safe=False, seed=None, seedw=0, tm_port=8000):

    args = argparse.Namespace(asynch=asynch, car_lights_on=car_lights_on, filterv=filterv, filterw=filterw,
                              generationv=generationv, generationw=generationw, hero=hero, host=host, hybrid=hybrid,
                              no_rendering=no_rendering, number_of_vehicles=number_of_vehicles, number_of_walkers=number_of_walkers, port=port, respawn=respawn,
                              safe=safe, seed=seed, seedw=seedw, tm_port=tm_port)

    thread = threading.Thread(target=__generate_traffic.main, args=[args])
    thread.run()

save_image_count = 0
def save_image(image):
    global save_image_count
    image_name = "camera_data/image%06d.png" % save_image_count
    image.save_to_disk(image_name)
    save_image_count += 1
    print(image_name)

camera_image = None
def save_image_memory(image):
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    camera_image = array

def show_image_cv2():
    global camera_image
    if camera_image is None:
        return
    cv2.imshow("camera", cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(2)

counter = 0
def count():
    global counter
    counter += 1
    print(counter)

