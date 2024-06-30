# Authors : 
#     @ Ahmet Emin Kazan:
#         Github: https://github.com/Ahmetkazann
#     @ MAMADY CHEICK SOUARE
#         Github: https://github.com/mamadycheicksouare
#     @ Carla simulator version: CARLA_0.9.15
#     @ Python version: Python 3.7.16
#     @ Virtual machine highly recommended...
#     @ Virtual machine: Anaconda Navigator
#     @ YOLOv8 For object detection
#     @ Reinforcement Learning : Stable Baselines3 PPO agent
#     @ Labeled Images with ROBOFLOW
#     @ YOLO Trained At Google Colab

import carla
import random
import gym
from gym import spaces
import pygame
import numpy as np
from global_route_planner import GlobalRoutePlanner # GPS
import time
from gym.spaces import Box
import cv2
import time
import math
from ultralytics import YOLO

ROTATE = 15 # Rotanin her zaman ileride olmamasi icin arabayi 15 derece sola veya saga donduruyoruz
# dondurmedigimiz senaryoda ajanımızı her seferinde duz giderek puan kasacaktır. Yani surekli ayni aksiyonu alarak EZBER yapacaktir.

class CarlaEnv(gym.Env):
    
    def __init__(self):
        super(CarlaEnv, self).__init__()
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        self.client.load_world('Town10HD_Opt')
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle_bp = self.world.get_blueprint_library().filter('*vehicle.lincoln.mkz_2020*')
        self.trafficobjectdetect_model = YOLO('final_best.pt') # egittigimiz model
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.sampling_resolution = 1
        self.width = 640
        self.height = 480
        
        self.action_space = spaces.Discrete(9) # throttle : 2, brake : 2, right : 2, left : 2, DO_NOTHING : 1

        self.angle_space = spaces.Box(low=np.array([-180.0]), high=np.array([180.0]), dtype=np.float32)
        self.visual_space = spaces.Box(low=0.0, high=1.0, shape=(self.height, self.width, 3), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'angle': self.angle_space,
            'image': self.visual_space
        })

    def gps(self,gps_start_location):
        grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)

        distance = 0
        for loc in self.spawn_points:

            cur_route = grp.trace_route(gps_start_location, loc.location)
            if len(cur_route)>distance:
                distance = len(cur_route)
                route = cur_route
        for waypoint in route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0), life_time=15.0,
                persistent_lines=True)
        navigation = []
        for r in range(len(route)):
            if r % 20 == 0:
                navigation.append(route[r])
        for n in range(len(navigation)):
            self.world.debug.draw_string(navigation[n][0].transform.location, str(n*20), draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0), life_time=25.0,
            persistent_lines=True)
        return navigation
    
    def angle_between(self, v1, v2):
        return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

    def get_angle(self, car,wp):

        vehicle_pos = car.get_transform()
        car_x = vehicle_pos.location.x
        car_y = vehicle_pos.location.y
        wp_x = wp[0].transform.location.x
        wp_y = wp[0].transform.location.y
        
        # vector to waypoint
        x = (wp_x - car_x)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
        y = (wp_y - car_y)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
        
        #car vector
        car_vector = vehicle_pos.get_forward_vector()
        degrees = self.angle_between((x,y),(car_vector.x,car_vector.y))
        if degrees > 180:
            degrees = degrees - 360 # normalize degree between 180, -180 if waypoint at leftside returns - degree otherwise +
        return degrees
        
    def collision_data(self, event):
        self.collision_hist.append(event)

        
    def reset(self): # cevre'yi tekrar egitime uygun hale getiriyoruz
        
        self.collision_hist = []
        self.reward = 0
        self.degree = 0
        self.angle = np.array([0.0], dtype=np.float32)
        self.destroy_actors() # clean actors
        self.actor_list = []
        for actor in self.actor_list:
            actor.destroy()
        
        r = random.randint(1,40)
        self.start_point = self.spawn_points[r]
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp[0], self.start_point)
        while self.vehicle == None:
            r = random.randint(1,40)
            self.start_point = self.spawn_points[r]
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp[0], self.start_point)
        time.sleep(0.5)
        angle_adj = random.randrange(-ROTATE, ROTATE, 1)
        trans = self.vehicle.get_transform()
        trans.rotation.yaw = trans.rotation.yaw + angle_adj
        self.vehicle.set_transform(trans)
        
        self.mycar_location = self.start_point.location # Ajanımızın bulundugu konumdan gps rotasini ciziyoruz
        self.navigation = self.gps(self.mycar_location)

        # collision sensor
        collision_transform = carla.Transform(carla.Location(z=1.3,x=1.4))
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, collision_transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        
        self.spectator = self.world.get_spectator()
        
        self.spectator.set_transform(carla.Transform(self.mycar_location + carla.Location(z=35),
        carla.Rotation(pitch=-90)))
        
        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.spectator)
        
        # RGB camera
        self.initial_location = self.vehicle.get_location()
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.height}")
        self.rgb_cam.set_attribute("fov", f"90")

		
        camera_init_trans = carla.Transform(carla.Location(z=2,x=3))
        self.sensor = self.world.spawn_actor(self.rgb_cam, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.camera_data = {'image': np.zeros((self.height, self.width, 4))}
        self.sensor.listen(lambda data: self.object_detect(data)) # cubed = lambda x: x ** 3
        time.sleep(5) # important one dont delete :)
        cv2.namedWindow('What RL Sees',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('What RL Sees', self.front_camera)
        cv2.waitKey(1)
        
        while self.front_camera is None:
            time.sleep(0.01)

        done = False

        self.world.tick()

        
        return {'angle': self.angle, 'image': self.front_camera}
    
    def object_detect(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((self.height, self.width, 4))[:, :, :3] # RGBA -> RGB
        
        results = self.trafficobjectdetect_model(i) # YOLO ile egittigimiz ARABA ve Trafik isigi objelerini algılayan model
        
        for result in results:
            boxes = result.boxes
            break
        box_screen = np.zeros((480,640,3), dtype=np.uint8)
        bbox = boxes.xyxy.tolist()
        bbox = [[int(element) for element in sublist] for sublist in bbox]
        for (xmin, ymin, xmax, ymax) in bbox:
            cv2.rectangle(box_screen, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=-1)

        roi = cv2.bitwise_and(box_screen,i)
        self.rgb_camera = i
        self.front_camera = roi
 
    def handle_navigation(self,vehicle): # Stack veri yapisi gibi calisir varmak istedigimiz lokasyonun rotasini cizeriz ve her 20 nokta self.navigation degiskeninde bulunur
        # self.navigation dizisinin 0. indisi ajanimiza en yakin waypointin verilerini tutar. 0. indise yeterince(8) yaklastigimizda diziden sileriz ve dizinin 1. indisi
        # 0. indisi haline gelir bu adım tekrarlanarak sehirdeki rotamizi adim adim takip etmeyi ögrenir.
        vehicle_x = self.vehicle.get_transform().location.x
        vehicle_y = self.vehicle.get_transform().location.y
        
        if math.sqrt((vehicle_x - self.navigation[0][0].transform.location.x) ** 2 + (vehicle_y - self.navigation[0][0].transform.location.y) ** 2) < 8: # yeterince yakin
            self.navigation.pop(0)
    def step(self, action):

        cam = self.front_camera
        rgbcam = self.rgb_camera
		# showing image
        cv2.imshow('What RL Sees', cam) # RGB kameramizda RL algoritmasinin yalnizca tepki vermesini istedigimiz objeleri cizerek egitimi hizlandırmak amaclanmistir
        # Algoritmanin gözlem uzayi: (varmak istedigimiz nokta ile gittigimiz dogrultu arasindaki aci (0 a ne kadar yakin o kadar iyi), "cam" degiskeni)
        cv2.imshow('Rgb Camera', rgbcam)
        cv2.waitKey(1)
        self.spectator.set_transform(carla.Transform(self.vehicle.get_transform().location + carla.Location(z=35),
        carla.Rotation(pitch=-90)))
        self.handle_navigation(self.vehicle)
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        if kmh < 10: # sabit durup puan kasmaması icin
            self.reward -= 10
        print(self.reward)
        self.degree = self.get_angle(self.vehicle, self.navigation[0])
        
        if len(self.collision_hist) != 0:
            done = True
            self.reward -= 3500
            self.destroy_actors()
        if abs(int(self.degree))< 10:
            self.reward += 3
        else:  
            self.reward -= (abs(int(self.degree)) / 10)  
            # varmak istedigimiz nokta ile arabanın dogrultusu arasındaki aci degeri ne kadar fazla ise o kadar ceza

        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.9, steer=0.0)) # w # tam gaz
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, brake=0)) # w # gaz
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=-0.5)) # d # sag
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=-0.75)) # d # tam sag
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer = 0.5)) # a # sol
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer = 0.75)) # a # tam sol
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 0.5)) # s # tam fren
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 0.25)) # s # fren
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 0.0)) # do nothing
            
        self.world.tick()
        done = self.is_done(self.reward)
        observation_dict = {'angle': self.angle, 'image': self.front_camera}
        return observation_dict, self.reward, done, {}
            
    def is_done(self,reward):
        if reward < - 3000:
            return True
        return False
        
    def destroy_actors(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()
        



