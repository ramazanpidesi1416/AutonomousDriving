from CarlaEnvironments import *

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

with CarlaEnvironment(port=4242, delta_seconds=1/30, no_rendering_mode=True) as env:
    env.change_map("town03")
    vehicle = Vehicle(env, "model3", True, 0)
    camera = Camera(env, vehicle.actor, IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    gnss = GNSS(env, vehicle.actor)

    for i in range(30):
        Vehicle(env, None, True, None)

    for i in range(30):
        Pedestrian(env, None)

    while True:
        begin = time.time()
        env.step()
        camera.display_data()
        gnss.display_data()
        #print("frame time: {:.2f}ms, simulation time: {:.1f} seconds".format((time.time()-begin)*1000, env.simulated_time))

