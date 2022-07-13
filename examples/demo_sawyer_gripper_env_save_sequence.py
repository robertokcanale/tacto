# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import gym
import sawyer_gripper_env  # noqa: F401
from numpy import concatenate
import cv2 
import os
class GraspingPolicy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.t = 0

    def forward(self, states=None):
        action = self.env.action_space.new()
        if not states:
            return action

        z_low, z_high = 0.15, 0.4
        dz = 0.02
        w_open, w_close = 0.11, 0.01
        gripper_force = 30

        if self.t < 50:
            action.end_effector.position = states.object.position + [0, 0, z_high]
            action.end_effector.orientation = [0.0, 0.5, 0.0, 0.0]
            action.gripper_width = w_open
        elif self.t < 100:
            s = (self.t - 50) / 50
            z = z_high - s * (z_high - z_low)
            action.end_effector.position = states.object.position + [0, 0, z-0.1]
        elif self.t < 150:
            action.gripper_width = w_close
            action.gripper_force = gripper_force
        elif self.t < 220:
            delta = [0, 0, dz]
            action.end_effector.position = states.robot.end_effector.position + delta
            action.gripper_width = w_close
            action.gripper_force = gripper_force
        else:
            action.gripper_width = w_close

        self.t += 1

        return action


def main():
    env = gym.make("sawyer-gripper-v0")
    env.reset()
    # Create a hard-coded grasping policy
    policy = GraspingPolicy(env)

    # Set the initial state (obs) to None, done to False
    obs, done = None, False
    print(os.getcwd())
    i = 0
    while not done:
        color, depth = env.render()
        action = policy(obs)
        obs, reward, done, info = env.step(action)

        colors = concatenate(color, axis=1)        
        depths =concatenate(list(map(env.digits._depth_to_color, depth)), axis=1)  
        cv2.imwrite(os.getcwd() + '/images/mug/color_' + str(i) + '.png',cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.getcwd() + '/images/mug/depth_' + str(i) + '.png',cv2.cvtColor(depths, cv2.COLOR_RGB2BGR))
        i = i+1
    env.close()


if __name__ == "__main__":
    main()
