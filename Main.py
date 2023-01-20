# default carla scripts
import cv2

import __generate_traffic as traffic
from CarlaEnvironments import *

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


env = CarlaEnvironment()
try:
    vehicle = Vehicle(env, "model3", True, 0)

    vehicles = []
    for i in range(30):
         vehicles.append(Vehicle(env, "model3", True))

    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    # vehicle.apply_control(throttle=1.0)

    camera.actor.listen(lambda image: save_image_memory(image))

    for i in range(3000):
        env.step()
        show_image_cv2()

finally:
    env.clear_objects()  # deletes all actors created during the execution of this script.
