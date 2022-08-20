# Code based on Carla examples, which are authored by 
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).

import carla
import random
from pathlib import Path
import numpy as np
import pygame
from util.carla_util import *
from util.geometry_util import dist_point_linestring
import argparse
import cv2


def get_trajectory_from_lane_detector(lane_detector, image):
    # get lane boundaries using the lane detector
    image_arr = carla_img_to_array(image)

    poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
    # https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
    img = img_left + img_right
    img = cv2.normalize(img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (600,400))
    
    # trajectory to follow is the mean of left and right lane boundary
    # note that we multiply with -0.5 instead of 0.5 in the formula for y below
    # according to our lane detector x is forward and y is left, but
    # according to Carla x is forward and y is right.
    x = np.arange(-2,60,1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    # hence correct x coordinates
    x += 0.5
    trajectory = np.stack((x,y)).T
    return trajectory, img

def get_trajectory_from_map(CARLA_map, vehicle):
    # get 80 waypoints each 1m apart. If multiple successors choose the one with lower waypoint.id
    waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    # transform waypoints to vehicle ref frame
    trajectory = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    trajectory = trajectory.T
    trajectory = trajectory[:,:2]
    return trajectory

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)



def main(fps_sim, mapid, weather_idx, showmap, model_type):
    # Imports
    from lane_detection.openvino_lane_detector import OpenVINOLaneDetector
    from lane_detection.lane_detector import LaneDetector
    from lane_detection.camera_geometry import CameraGeometry
    from control.pure_pursuit import PurePursuitPlusPID

    actor_list = []
    pygame.init()

    display, font, clock, world = create_carla_world(pygame, mapid)

    weather_presets = find_weather_presets()
    world.set_weather(weather_presets[weather_idx][0])

    controller = PurePursuitPlusPID()
    cross_track_list = []
    fps_list = []

    try:
        CARLA_map = world.get_map()

        # create a vehicle
        blueprint_library = world.get_blueprint_library()
        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','64,81,181')
        spawn_point = random.choice(CARLA_map.get_spawn_points())

        vehicle = world.spawn_actor(veh_bp, spawn_point)
        actor_list.append(vehicle)

        # Show map
        if showmap:
            plot_map(CARLA_map, mapid, vehicle)

        startPoint = spawn_point
        startPoint = carla_vec_to_np_array(startPoint.location)

        # visualization cam (no functionality)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]
        
        # Lane Detector Model
        # ---------------------------------
        cg = CameraGeometry()
        
        # TODO: Change model here
        if model_type == "openvino":
            lane_detector = OpenVINOLaneDetector()
        else:
            lane_detector = LaneDetector(model_path=Path("lane_detection/Deeplabv3+(MobilenetV2).pth").absolute())

        # Windshield cam
        cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        bp = blueprint_library.find('sensor.camera.rgb')
        fov = cg.field_of_view_deg
        bp.set_attribute('image_size_x', str(cg.image_width))
        bp.set_attribute('image_size_y', str(cg.image_height))
        bp.set_attribute('fov', str(fov))
        camera_windshield = world.spawn_actor(
            bp,
            cam_windshield_transform,
            attach_to=vehicle)
        actor_list.append(camera_windshield)
        sensors.append(camera_windshield)
        # ---------------------------------

        flag = True
        max_error = 0
        FPS = fps_sim

        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                
                # Advance the simulation and wait for the data. 
                tick_response = sync_mode.tick(timeout=2.0)

                snapshot, image_rgb, image_windshield = tick_response
                try:
                    trajectory, img = get_trajectory_from_lane_detector(lane_detector, image_windshield)
                except:
                    trajectory = get_trajectory_from_map(CARLA_map, vehicle)

                max_curvature = get_curvature(np.array(trajectory))
                if max_curvature > 0.005 and flag == False:
                    move_speed = np.abs(25 - 100*max_curvature)
                else:
                    move_speed = 25

                speed = np.linalg.norm( carla_vec_to_np_array(vehicle.get_velocity()))
                throttle, steer = controller.get_control(trajectory, speed, desired_speed=move_speed, dt=1./FPS)
                send_control(vehicle, throttle, steer, 0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                dist = dist_point_linestring(np.array([0,0]), trajectory)

                cross_track_error = int(dist)
                max_error = max(max_error, cross_track_error)
                if cross_track_error > 0:
                    cross_track_list.append(cross_track_error)
                waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(waypoint.transform.location)

                if np.linalg.norm(vehicle_loc-startPoint) > 20:
                    flag = False

                if np.linalg.norm(vehicle_loc-startPoint) < 20 and flag == False:
                    print('done.')
                    break
                
                if speed < 1 and flag == False:
                    print("----------------------------------------\nSTOP, car accident !!!")
                    break

                fontText = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.75
                fontColor = (255,255,255)
                lineType = 2
                laneMessage = "No Lane Detected"
                steerMessage = ""
                
                if dist < 0.75:
                    laneMessage = "Lane Tracking: Good"
                else:
                    laneMessage = "Lane Tracking: Bad"

                cv2.putText(img, laneMessage,
                        (350,50),
                        fontText,
                        fontScale,
                        fontColor,
                        lineType)             

                if steer > 0:
                    steerMessage = "Right"
                else:
                    steerMessage = "Left"

                cv2.putText(img, "Steering: {}".format(steerMessage),
                        (400,90),
                        fontText,
                        fontScale,
                        fontColor,
                        lineType)

                steerMessage = ""
                laneMessage = "No Lane Detected"

                cv2.putText(img, "X: {:.2f}, Y: {:.2f}".format((vehicle_loc[0]), vehicle_loc[1], vehicle_loc[2]),
                            (20,50),
                            fontText,
                            0.5,
                            fontColor,
                            lineType)

                cv2.imshow('Lane detect', img)
                cv2.waitKey(1)

                fps_list.append(clock.get_fps())

                # Draw the display pygame.
                draw_image(display, image_rgb)
                display.blit(
                    font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('     FPS (simulated): % 5d ' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('     speed: {:.2f} km/h'.format(speed*3.6), True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('     cross track error: {:03d} m'.format(cross_track_error*100), True, (255, 255, 255)),
                    (8, 64))
                display.blit(
                    font.render('     max cross track error: {:03d} m'.format(max_error), True, (255, 255, 255)),
                    (8, 82))

                pygame.display.flip()


    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        print('mean cross track error: ',np.mean(np.array(cross_track_list)))
        print('mean fps: ',np.mean(np.array(fps_list)))
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Carla simulation with your control algorithm.')
    parser.add_argument("--mapid", default = "4", help="Choose map from 1 to 5")
    parser.add_argument("--fps", type=int, default=20, help="Setting FPS")
    parser.add_argument("--weather", type=int, default=6, help="Check function find_weather in carla_util.py for mor information")
    parser.add_argument("--showmap", type=bool, default=False, help="Display Map")
    parser.add_argument("--model", default="openvino", help="Choose between OpenVINO model and PyTorch model")
    args = parser.parse_args()

    try:
        main(fps_sim = args.fps, mapid = args.mapid, weather_idx=args.weather, showmap=args.showmap, model_type=args.model)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
