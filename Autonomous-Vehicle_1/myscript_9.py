import carla
import random
import gym
from gym import spaces
import pygame
import numpy as np
from global_route_planner import GlobalRoutePlanner
import time

from gym.spaces import Box

import cv2

import time
import math

# client = carla.Client('localhost', 2000)
# client.set_timeout(20.0)
# world = client.get_world()
# spawn_points = world.get_map().get_spawn_points()
#vehicle_bp = world.get_blueprint_library().filter('*vehicle.lincoln.mkz_2020*')
#vehicle_bp = world.get_blueprint_library().filter('*mini*')
ROTATE = 15

#print(spawn_points[0])
class CarlaEnv(gym.Env):
    
    def __init__(self):
        super(CarlaEnv, self).__init__()
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        self.client.load_world('Town10HD_Opt')
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle_bp = self.world.get_blueprint_library().filter('*vehicle.lincoln.mkz_2020*')
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.sampling_resolution = 1
        self.current_time = time.time()
        self.gps_timer = time.time()
        
        self.action_space = spaces.Discrete(9) # throttle, brake, right, left, DO_NOTHING
        # self.observation_space = spaces.Box(low=0.0, high=1.0,
        #         shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
        #self.observation_space = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))
        self.observation_space = Box(low=np.array([-180.0]), high=np.array([180.0]), dtype=np.float32)

        
        
    def gps(self,gps_start_location):
        grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)

        distance = 0
        for loc in self.spawn_points: # we start trying all spawn points 
                                    #but we just exclude first at zero index
            cur_route = grp.trace_route(gps_start_location, loc.location)
            if len(cur_route)>distance:
                distance = len(cur_route)
                route = cur_route
        #draw the route in sim window - Note it does not get into the camera of the car
        for waypoint in route:
            self.world.debug.draw_string(waypoint[0].transform.location, '^', draw_shadow=False,
                color=carla.Color(r=0, g=0, b=255), life_time=15.0,
                persistent_lines=True)
        navigation = []
        for r in range(len(route)):
            if r % 20 == 0:
                navigation.append(route[r])
        for n in range(len(navigation)):
            self.world.debug.draw_string(navigation[n][0].transform.location, str(n*20), draw_shadow=False,
            color=carla.Color(r=0, g=0, b=255), life_time=25.0,
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
    def reset(self):
        self.collision_hist = []
        self.reward = 0
        self.degree = 0
        self.destroy_actors() # clean stuff
        self.actor_list = []
        for actor in self.actor_list:
            actor.destroy()
        
        print(len(self.spawn_points))
        
        r = random.randint(1,20)
        self.start_point = self.spawn_points[r]
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp[0], self.start_point)
        while self.vehicle == None:
            r = random.randint(1,20)
            self.start_point = self.spawn_points[r]
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp[0], self.start_point)
        time.sleep(0.5)
        angle_adj = random.randrange(-ROTATE, ROTATE, 1)
        trans = self.vehicle.get_transform()
        #trans = self.start_point
        trans.rotation.yaw = trans.rotation.yaw + angle_adj
        self.vehicle.set_transform(trans)
        
        
        print("araba cikti",r)
        self.mycar_location = self.start_point.location #we start at where the car is
        #transform = self.vehicle.get_transform()
        self.navigation = self.gps(self.mycar_location)
        
        # collision
        collision_transform = carla.Transform(carla.Location(z=1.3,x=1.4))
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, collision_transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        #
        
        
        self.spectator = self.world.get_spectator()
        
        self.spectator.set_transform(carla.Transform(self.mycar_location + carla.Location(z=50),
        carla.Rotation(pitch=-90)))
        
        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.spectator)
        #
        #bp_lib = world.get_blueprint_library()
        # camera_bp = bp_lib.find('sensor.camera.rgb')
        # camera_init_trans = carla.Transform(carla.Location(x = 3,z=2))
        # camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to = self.vehicle)
        # time.sleep(0.5)
        
        # image_w = camera_bp.get_attribute("image_size_x").as_int()
        # image_h = camera_bp.get_attribute("image_size_y").as_int()
        
        # camera_data = {'image': np.zeros((image_h, image_w, 4))}
    
        #camera.listen(lambda image: self.camera_callback(image, camera_data))
        # cv2.namedWindow('RGBCamera', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RGBCamera', camera_data['image'])
        # cv2.waitKey(1)
        # done = self._is_done()
        done = False
        #self.gps(self.mycar_location) # draw a route from car location to furthest location
        #while not done:
            #rgb_image = camera_data
            #cv2.imshow('RGBCamera', rgb_image)
            #cv2.imshow('RGBCamera', camera_data)
            
            
        self.world.tick()
        # while self.camera_image is None:
        #     self.world.tick()
        #
        
        return self.degree

    # def _is_done(self):
    #     if self._check_collision():
    #         print("collision")
    #         return True
    #     if self.vehicle.get_location().z < 0:  # Araç haritanın dışına düştüyse
    #         print("FALL")
    #         return True
    #     return False
    
    def camera_callback(self, image, data_dict):
            data_dict = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) # sonra bakılacak
            
    def handle_navigation(self,vehicle):
        vehicle_x = self.vehicle.get_transform().location.x
        vehicle_y = self.vehicle.get_transform().location.y
        
        if math.sqrt((vehicle_x - self.navigation[0][0].transform.location.x) ** 2 + (vehicle_y - self.navigation[0][0].transform.location.y) ** 2) < 8:
            self.navigation.pop(0)
            print("silinen navigasyon",len(self.navigation))
            print("0.indis",self.navigation[0][0].transform.location.x)
    def step(self, action):
        
        self.spectator.set_transform(carla.Transform(self.vehicle.get_transform().location + carla.Location(z=50),
        carla.Rotation(pitch=-90)))
        self.handle_navigation(self.vehicle)
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # if kmh < 10: # sabit durup puan kasmaması icin
        #     print("kmh",kmh)
        #     self.reward -= 5
        
        self.ilerleme_istegi = (abs(self.vehicle.get_transform().get_forward_vector().x) + abs(self.vehicle.get_transform().get_forward_vector().y) + abs(self.vehicle.get_transform().get_forward_vector().y)) / 3
        
        if self.ilerleme_istegi < 0.5:
            print("ilerleme istegi dusuk ceza !!")
            self.reward -= 3
        print("ilerleme istegi:", self.ilerleme_istegi)
        self.degree = self.get_angle(self.vehicle, self.navigation[0])
        
        if len(self.collision_hist) != 0:
            done = True
            self.reward -= 1000
            self.destroy_actors()
        if abs(int(self.degree))< 10:
            self.reward += 1
        else:  
            self.reward -= (abs(int(self.degree)) / 10)
            print("acidegerireward",(abs(int(self.degree)) / 10))
        
        # if abs(self.degree) < 10:
        #     self.reward += 3
        # elif abs(self.degree) < 30:
        #     self.reward += 2
        # elif abs(self.degree) < 60:
        #     self.reward += 1
        # else:
        #     self.reward -= 3
        
        #print(self.reward)
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
        #print("rew",self.reward)
        done = self.is_done(self.reward)
        
        return self.degree, self.reward, done, {}
            
        #self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer, brake = 0.0))
    
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
        



