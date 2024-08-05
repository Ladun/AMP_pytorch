import gymnasium
import tqdm
import numpy as np
from numpngw import write_apng
from collections import deque

import custom_env
from algo.utils.general import TimerManager
from algo.data.preprocess_data import parse_amc

def get_inference_video(frame, total_time, vid_name="render.png"):
    timestep = 1. / frame
    images = []
    timer_manager = TimerManager()
    env =  gymnasium.make("HumanoidBulletEnv-v0", env_config={"render_mode":"rgb_array"})
    env.reset()

    try:
        env.reset()
        with timer_manager.get_timer("Total"):
            for i in tqdm.tqdm(range(frame * total_time)):
                with timer_manager.get_timer("one step"):
                    env.step(np.random.uniform(env.robot.action_space.low,
                                               env.robot.action_space.high,
                                               env.robot.action_space.shape))

                with timer_manager.get_timer("one render"):
                    img = env.render()
                images.append(img)
        with timer_manager.get_timer("create images"):
            write_apng(vid_name, images, delay=timestep * 1000)
    except Exception as e:
        env.close()
        print(e)
        
    for k, v in timer_manager.timers.items():
        print(f"\t\t {k} time: {v.get()} sec")

    return images

def env_test():
    from getkey import getkey, keys
    
    timer_manager = TimerManager()
    env =  gymnasium.make("HumanoidBulletEnv-v0", env_config={"render_mode":"human"})
    env.reset()

    try:
        env.reset()
        while True:
            with timer_manager.get_timer("one step"):
                env.step(np.random.uniform(env.robot.action_space.low,
                                            env.robot.action_space.high,
                                            env.robot.action_space.shape))

            with timer_manager.get_timer("one render"):
                img = env.render()
            
            key = getkey(blocking=True)
            print(key)
    except Exception as e:
        env.close()
        print(e)
        
    for k, v in timer_manager.timers.items():
        print(f"\t\t {k} time: {v.get()} sec")


# images = get_inference_video(240, 1)
# print(len(images))

# env_test()

class Buffer:
    def __init__(self):
        self.a = []
        self.b = []
        
    def add(self, **d):
        for k, v in d.items():
            self.__dict__[k].append(v)
        

b = Buffer()

b.add(a=1,b=2)

print(b.a, b.b)