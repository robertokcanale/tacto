# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import cv2
import os
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO

log = logging.getLogger(__name__)
# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT
    digit_body = px.Body(**cfg.digit)
    digits.add_camera(digit_body.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    print(cfg.object)
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()
    log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    
    

    
    
    #Simulation Start 
    t.start()
    color, depth = digits.render()
    #preparing the iarrays mages as an image, for coloured and depth image
    colors = np.concatenate(color, axis=1)        
    depths =np.concatenate(list(map(digits._depth_to_color, depth)), axis=1)

    # Convert to gray scale
    prvs = cv2.cvtColor(depths, cv2.COLOR_BGR2GRAY)
    # Create mask
    hsv_mask = np.zeros_like(depths)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255

    while True:
        #Getting sensor data as a single array        
        color, depth = digits.render()
        
        #preparing the iarrays mages as an image, for coloured and depth image
        colors = np.concatenate(color, axis=1)        
        depths =np.concatenate(list(map(digits._depth_to_color, depth)), axis=1)

        next = cv2.cvtColor(depths, cv2.COLOR_BGR2GRAY)
    
        # Optical flow is now calculated
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnite and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to rgb
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        
        
        resized = cv2.resize(rgb_representation, (400,400), interpolation = cv2.INTER_AREA)
        # Opens a new window and displays the output frame
        cv2.imshow("dense optical flow", resized)
        
        # Updates previous frame
        prvs = next
        
        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        digits.updateGUI(color, depth)

    t.stop()


        
if __name__ == "__main__":
    main()
