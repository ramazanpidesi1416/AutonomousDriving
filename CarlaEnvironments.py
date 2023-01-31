import math

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
import keyboard

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

    def __init__(self, delta_seconds=1/30.0, no_rendering_mode=False, synchronous_mode=True, port=2000):
        self.initialization_successful = False
        self.deleted = False
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.delta_seconds = delta_seconds
        self.no_rendering_mode = no_rendering_mode
        self.synchronous_mode = synchronous_mode
        self._settings = None
        self.frame = None
        self.actor_list = []
        self.simulated_time = 0
        self.simulated_step_count = 0
        self.step_time = self.delta_seconds

        self.connect_to_host(port)
        self.initialize_world()
        #self.change_map("town03")
        self.clear_objects()

        self.initialization_successful = True

        print("scene initialization is successful")

    def connect_to_host(self, port):
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)


    def initialize_world(self):
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=self.no_rendering_mode,
            synchronous_mode=self.synchronous_mode,
            fixed_delta_seconds=self.delta_seconds))

    def change_map(self, map_name="town3"):
        self.world = self.client.load_world(map_name)
        self.initialize_world()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__del__()

    def __del__(self):
        print("destructor is called")
        self.change_settings(delta_seconds=1 / 30.0, no_rendering_mode=False, synchronous_mode=False)
        self.clear_objects()

    def clear_objects(self):
        for a in self.world.get_actors().filter("vehicle*"):
            if a.is_alive:
                try:
                    a.destroy()
                except Exception as e:
                    print(e)

        for a in self.world.get_actors().filter("walker.pedestrian*"):
            if a.is_alive:
                try:
                    a.destroy()
                except Exception as e:
                    print(e)

        for a in self.world.get_actors().filter("controller.ai.walker"):
            if a.is_alive:
                try:
                    a.destroy()
                except Exception as e:
                    print(e)

        for a in self.world.get_actors().filter("sensor*"):
            if a.is_alive:
                try:
                    a.destroy()
                except Exception as e:
                    print(e)

    def step(self, print_step_time=False):
        begin = time.time()
        self.frame = self.world.tick()
        self.simulated_time += self.delta_seconds
        self.simulated_step_count += 1
        self.step_time = (time.time()-begin)*1000

        if print_step_time:
            print("frame time: {:.2f}ms, simulation time: {:.1f} seconds".format(self.step_time, self.simulated_time))


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

    def __add__(self, other):
        return vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return vector(other * self.x, other * self.y, other * self.z)

    def __repr__(self):
        return "x:{:.2f} y:{:.2f} z:{:.2f}".format(self.x, self.y, self.z)

    def length(self):
        return math.sqrt((self.x*self.x) + (self.y*self.y) + (self.z*self.z))

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
                self.bp = carla_environment.blueprint_library.filter(vehicle_type)[0]
            elif vehicle_type is None:
                self.bp = random.choice(carla_environment.blueprint_library.filter('vehicle.*.*'))
            else:
                print("vehicle type must be a string or None")
                raise Exception

            self.actor = carla_environment.world.spawn_actor(self.bp, spawn_point)
        except Exception:
            print("vehicle couldn't initialized")
            return

        carla_environment.actor_list.append(self.actor)
        self.autopilot = autopilot
        self.actor.set_autopilot(self.autopilot)

        self.total_distance_travelled = 0.0
        self._last_position = None
        self._ignore_update = 5     # ignore the first 5 update of the simulation

    def get_position(self):
        return vector(self.actor.get_transform().location.x, self.actor.get_transform().location.y, self.actor.get_transform().location.z)

    def get_rotation(self):
        return vector(self.actor.get_transform().rotation.pitch, self.actor.get_transform().rotation.yaw, self.actor.get_transform().rotation.roll)

    def update_total_distance_travelled(self):
        current_position = self.get_position()
        if self._last_position is not None and self._ignore_update == 0:
            self.total_distance_travelled += (current_position - self._last_position).length()
        if self._ignore_update > 0:
            self._ignore_update -= 1

        self._last_position = current_position

    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0):
        self.actor.apply_control(carla.VehicleControl(throttle=throttle,
                                                      steer=steer,
                                                      brake=brake,
                                                      hand_brake=hand_brake,
                                                      reverse=reverse,
                                                      manual_gear_shift=manual_gear_shift,
                                                      gear=gear))

    def update_manuel_control(self):
        throttle = 0
        brake = 0
        steer = 0
        if keyboard.is_pressed('ı'):
            throttle = 0.5
        if keyboard.is_pressed('k'):
            brake = 0.3
        if keyboard.is_pressed('j'):
            steer = -0.3
        if keyboard.is_pressed('l'):
            steer = 1
        self.apply_control(throttle=throttle, steer=steer, brake=brake)


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

    def __init__(self, carla_environment, attaching_carla_actor, camera_type="rgb", image_width=640, image_height=480, fov=110, displacement=vector(2.5, 0, 0.7)):
        self.blueprint = None
        self.displacement = None
        self.actor = None
        self.camera_type = None

        self.initialize_camera(carla_environment, attaching_carla_actor, camera_type, image_width, image_height, fov, displacement)

        self.camera_image = None
        self.actor.listen(lambda image: self.save_image_memory(image))
        self.save_image_disc_count = 0

    def initialize_camera(self, carla_environment, attaching_carla_actor, camera_type, image_width, image_height, fov, displacement):
        if self.actor is not None:
            carla_environment.actor_list.remove(self.actor)
            self.actor.destroy()

        self.camera_type = camera_type
        if camera_type == "rgb":
            self.blueprint = carla_environment.blueprint_library.find('sensor.camera.rgb')
        elif camera_type == "semantic":
            self.blueprint = carla_environment.blueprint_library.find('sensor.camera.semantic_segmentation')
        else:
            print("camera type is not valid, please use rgb or semantic camera")
            return

        self.blueprint.set_attribute("image_size_x", f"{image_width}")
        self.blueprint.set_attribute("image_size_y", f"{image_height}")
        self.blueprint.set_attribute("fov", f"{fov}")
        self.displacement = carla.Transform(carla.Location(displacement.x, displacement.y, displacement.z))
        self.actor = carla_environment.world.spawn_actor(self.blueprint, self.displacement, attach_to=attaching_carla_actor)
        carla_environment.actor_list.append(self.actor)

    def save_image_memory(self, carla_image):
        if self.camera_type == "semantic":
            carla_image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (carla_image.height, carla_image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_image = array

    def save_image_to_disc(self):
        if self.camera_image is None:
            print("can't save image. image is null")
            return
        image_name = "camera_data/image%06d.png" % self.save_image_disc_count
        cv2.imwrite(image_name, cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2RGB))
        self.save_image_disc_count += 1
        print("{image_name} is saved to disc".format(image_name=image_name))

    def display_data(self, window_name="camera"):
        if self.camera_image is None:
            return
        cv2.imshow(window_name, cv2.cvtColor(self.camera_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(2)

class GNSS:

    def __init__(self, carla_environment, attaching_carla_actor):
        self.blueprint = carla_environment.world.get_blueprint_library().find('sensor.other.gnss')
        self.transform = attaching_carla_actor.get_transform()
        self.actor = carla_environment.world.spawn_actor(self.blueprint, self.transform, attach_to=attaching_carla_actor)
        self.actor.listen(lambda data: self.save_data_memory(data))
        self.gnss_data = None
        carla_environment.actor_list.append(self.actor)

        self.data_position = None
        self.data_rotation = None

    def save_data_memory(self, data):
        self.gnss_data = data
        self.data_position = vector(self.gnss_data.transform.location.x, self.gnss_data.transform.location.y, self.gnss_data.transform.location.z)
        self.data_rotation = vector(self.gnss_data.transform.rotation.pitch, self.gnss_data.transform.rotation.yaw, self.gnss_data.transform.rotation.roll)

    def display_data(self):
        print("X:{:.2f} Y:{:.2f} Z:{:.2f} rotX={:.2f} rotY={:.2f} rotZ={:.2f}".format(self.data_position.x, self.data_position.y, self.data_position.z, self.data_rotation.x, self.data_rotation.y, self.data_rotation.z))


class CollusionSensor:

    def __init__(self, carla_environment, attaching_carla_actor):
        self.blueprint = carla_environment.world.get_blueprint_library().find('sensor.other.collision')
        self.transform = attaching_carla_actor.get_transform()
        self.actor = carla_environment.world.spawn_actor(self.blueprint, self.transform,
                                                         attach_to=attaching_carla_actor)
        self.actor.listen(lambda data: self.save_data_memory(data))
        self.collusion_history = []
        self.print_queue = []
        carla_environment.actor_list.append(self.actor)

    def save_data_memory(self, data):
        self.collusion_history.append(data)
        self.print_queue.append(data)

    def display_data(self):
        for collusion in self.print_queue:
            self.print_queue.remove(collusion)
            print(collusion)

class LaneInvasionSensor:

    def __init__(self, carla_environment, attaching_carla_actor):
        self.blueprint = carla_environment.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.transform = attaching_carla_actor.get_transform()
        self.actor = carla_environment.world.spawn_actor(self.blueprint, self.transform,
                                                         attach_to=attaching_carla_actor)
        self.actor.listen(lambda data: self.save_data_memory(data))
        self.lane_invasion_history = []
        self.print_queue = []

        carla_environment.actor_list.append(self.actor)

    def save_data_memory(self, data):
        self.lane_invasion_history.append(data)
        self.print_queue.append(data)

    def display_data(self):
        for Invasion in self.print_queue:
            self.print_queue.remove(Invasion)
            print(Invasion)
            for marking in Invasion.crossed_lane_markings:
                print(marking.type)
                print(marking.color)
                print(marking.lane_change)

class Lidar:

    def __init__(self, carla_environment, attaching_carla_actor):
        pass

    def save_data_memory(self, data):
        pass



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

counter = 0
def count():
    global counter
    counter += 1
    print(counter)

