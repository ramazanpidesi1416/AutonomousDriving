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

pygame_screen = None
def pygame_init():
    global pygame_screen
    if pygame_screen is None:
        pygame.init()
        pygame.display.set_caption("camera")
        pygame_screen = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT))

def show_image(image):
    global pygame_screen
    pygame_init()
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    pygame_screen.blit(image_surface, (0, 0))
    pygame.display.update()
    time.sleep(1/60)

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
    vehicle = Vehicle(env, "model3", False, 0)
    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))

    vehicle.apply_control(throttle=1.0)

    camera.actor.listen(lambda image: show_image(image))

    while True:
        env.step()

    #generate_traffic()


finally:
    env.__del__()  # deletes all actors created during the execution of this script.
