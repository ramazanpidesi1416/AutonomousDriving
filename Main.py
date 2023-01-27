from CarlaEnvironments import *
import keyboard

# hyper parameters
IM_WIDTH = 640
IM_HEIGHT = 480

with CarlaEnvironment(port=2000, delta_seconds=1/15, no_rendering_mode=False) as env:
    env.change_map("town03")
    vehicle = Vehicle(env, "model3", False, 0)
    camera = Camera(env, vehicle.actor, "rgb", IM_WIDTH, IM_HEIGHT, 110, vector(2.5, 0, 0.7))
    gnss = GNSS(env, vehicle.actor)
    collusion = CollusionSensor(env, vehicle.actor)
    lane_invasion = LaneInvasionSensor(env, vehicle.actor)

    #vehicle.apply_control(throttle=1)

    for i in range(30):
        Vehicle(env, None, True, None)

    for i in range(30):
        Pedestrian(env, None)

    while env.simulated_time < 200:     # 50 seconds in simulation-time
        throttle = 0
        brake = 0
        steer = 0
        if keyboard.is_pressed('ı') or keyboard.is_pressed('w'):
            throttle = 1
        if keyboard.is_pressed('k') or keyboard.is_pressed('s'):
            brake = 1
        if keyboard.is_pressed('j') or keyboard.is_pressed('a'):
            steer = -1
        if keyboard.is_pressed('l') or keyboard.is_pressed('d'):
            steer = 1
        vehicle.apply_control(throttle=throttle, steer=steer, brake=brake)
        begin = time.time()
        vehicle.update_total_distance_travelled()
        print(vehicle.total_distance_travelled)
        env.step()
        camera.display_data("camera1")
        # print("frame time: {:.2f}ms, simulation time: {:.1f} seconds".format((time.time()-begin)*1000, env.simulated_time))

