# default carla scripts
import __generate_traffic as traffic
from CarlaEnvironments import *

import matplotlib.pyplot as plt
import pygame

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

save_image_count = 0
def save_image(image):
    global save_image_count
    image_name = "camera_data/image%06d.png" % save_image_count
    image.save_to_disk(image_name)
    save_image_count += 1
    print(image_name)

def pygame_init():
    pygame.init()
    pygame.display.set_caption("camera")
    pygame_screen = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT))
    return pygame_screen


camera_image = None
def save_image_memory(image):
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    camera_image = array

def show_image_pygame(pygame_screen):
    global camera_image
    if camera_image is None:
        return
    image_surface = pygame.surfarray.make_surface(camera_image.swapaxes(0, 1))
    pygame_screen.blit(image_surface, (0, 0))
    pygame.display.update()
    pygame.event.pump()

counter = 0
def count():
    global counter
    counter += 1
    print(counter)


env = CarlaEnvironment()
try:
    vehicles = []
    #for i in range(30):
    #     vehicles.append(Vehicle(env, "model3", True))
    vehicle = Vehicle(env, "model3", True, 0)
    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))

    # vehicle.apply_control(throttle=1.0)

    camera.actor.listen(lambda image: save_image_memory(image))
    pygame_screen = pygame_init()

    while True:
        env.step()
        show_image_pygame(pygame_screen)

    #generate_traffic()


finally:
    env.__del__()  # deletes all actors created during the execution of this script.
