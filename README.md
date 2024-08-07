# In this project, it is aimed to train an Reinforcement Learning model that can go from point A to point B in the city in the carla simulator environment, obeying the traffic rules.
##![1](https://github.com/user-attachments/assets/e689033c-2bee-4762-923a-66e944b08ce4)
For this, we determine a random route through the city from the global route planner class and use the angle value that will enable it to follow every 20th point (10 meters) with the stack data structure for training our RL model.
##![2](https://github.com/user-attachments/assets/1634b12c-57d0-49e4-8a44-ac8af8c306ef)
When the car reaches each point, the point is deleted from the stack data structure and the next point is updated as observation space.


##![3](https://github.com/user-attachments/assets/b64d6a84-3d4f-4131-89d8-f81e0594a7f4)
In this way, it reaches point a from point b by following the route.

## The Observation space:
- Angle for route
- What RL SEES Screen
On this screen, the images detected for the car to comply with the traffic rules are transferred to the black screen. In this way, the RL model sees only the necessary objects, and the training is accelerated because the states are just that.
We trained yolo model for this objective. Model only detect car and traffic light.


## Rewards system:
- The greater the angle between the direction the car is going and the direction it should go, the more penalty it will receive.
- Any collision is penalized by -3000 and resets the training.


Used Simulations and Libraries
- Carla Simulator
- Stable_Baselines3
- Numpy
- Stable_Baselines3 PPO (Reinforcement Learning)
- OpenAI GYM Library
- Ultralytics YOLO
