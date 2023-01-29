from CarlaEnvironments import *

# hyper parameters
IM_WIDTH = 256
IM_HEIGHT = 256

batch_size = 1024

with CarlaEnvironment(port=4221, delta_seconds=1/60, no_rendering_mode=False) as env:
    env.change_map("town03")
    vehicle = Vehicle(env, "model3", True, 0)
    camera = Camera(env, vehicle.actor, "semantic", IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    gnss = GNSS(env, vehicle.actor)
    collusion = CollusionSensor(env, vehicle.actor)
    lane_invasion = LaneInvasionSensor(env, vehicle.actor)

    #vehicle.apply_control(throttle=1)

    for i in range(30):
        Vehicle(env, None, True, None)

    for i in range(30):
        Pedestrian(env, None)

    while True:     # 50 seconds in simulation-time
        #vehicle.update_total_distance_travelled()
        #print(vehicle.total_distance_travelled)
        #vehicle.update_manuel_control()
        env.step(True)
        camera.display_data("camera1")


