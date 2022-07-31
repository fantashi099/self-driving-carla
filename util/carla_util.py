import carla
import pygame

import os
import cv2
import queue
import random
import numpy as np

def carla_vec_to_np_array(vec):
    return np.array([vec.x,
                     vec.y,
                     vec.z])

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            no_rendering_mode=False,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def create_carla_world(pygame, mapid):
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(40.0)

    client.load_world('Town0' + mapid)
    world = client.get_world()
    return display, font, clock, world


def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def plot_map(m, mapid, vehicle = None):
    import matplotlib.pyplot as plt

    wp_list = m.generate_waypoints(2.0)
    loc_list = np.array(
        [carla_vec_to_np_array(wp.transform.location) for wp in wp_list]
    )
    plt.scatter(loc_list[:, 0], loc_list[:, 1])

    if vehicle != None:
        wp = m.get_waypoint(vehicle.get_transform().location)
        vehicle_loc = carla_vec_to_np_array(wp.transform.location)
        plt.scatter([vehicle_loc[0]], [vehicle_loc[1]])
        plt.title(f'Town0{mapid}')

    plt.show()


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    # print(curvature)
    return np.max(curvature)


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_img(image, path, raw=False):
    array = carla_img_to_array(image)
    if raw:
        np.save(path, array)
    else:
        cv2.imwrite(path, array)


def get_random_spawn_point(CARLA_map):
    pose = random.choice(CARLA_map.get_spawn_points())
    return CARLA_map.get_waypoint(pose.location)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def find_weather_presets():
    """
    presets: ['ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 
        'CloudyNoon', 'CloudySunset', 'Default', 'HardRainNight', 
        'HardRainNoon', 'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 
        'MidRainyNoon', 'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 
        'WetCloudyNight', 'WetCloudyNoon', 'WetCloudySunset', 'WetNight', 
        'WetNoon', 'WetSunset']
    
    return: [<Class Weather>, "Weather"] 
    E.g: [<Class ClearNight>, "ClearNight"]
    """
    import re
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    output = [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    return output