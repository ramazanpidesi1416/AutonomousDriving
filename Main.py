from CarlaEnvironments import *

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

image_count = 0
def save_image(image):
    global image_count
    image.save_to_disk("camera_data/image%06d.png" % image_count)
    image_count += 1
    print(image_count)

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

counter = 0
def count():
    global counter
    counter += 1
    print(counter)


env = CarlaEnvironment()
try:
    vehicle = Vehicle(env, "model3")
    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))

    vehicle.apply_control(throttle=1.0)

    camera.actor.listen(lambda image: save_image(image))

    while True:
        env.step()


finally:
    env.__del__()  # deletes all actors created during the execution of this script.
