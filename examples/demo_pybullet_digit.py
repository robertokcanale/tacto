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

image_resize = [400,400] #hopefully the x is larger than 100

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
    t.start()

    while True:
        #Getting sensor data as a single array        
        color, depth = digits.render()
        
        #preparing the iarrays mages as an image, for coloured and depth image
        colors = np.concatenate(color, axis=1)        
        depths =np.concatenate(list(map(digits._depth_to_color, depth)), axis=1)
       
       #resizing the images if needed
        depths = cv2.resize(depths, (400, 400), interpolation = cv2.INTER_AREA)
        colors = cv2.resize(colors, (400, 400), interpolation = cv2.INTER_AREA)
        #Showing the iamges
        cv2.imshow("depths", cv2.cvtColor(depths, cv2.COLOR_RGB2BGR))
        cv2.imshow("colors", cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))
        c = cv2.waitKey(2000)
        print (os.getcwd())        
        if (c == ord('s')): 
            print("saved")
            cv2.imwrite(os.getcwd()+'/contact_image_dataset/test_depths.jpeg',depths) 
            cv2.imwrite(os.getcwd()+'/contact_image_dataset/test_colors.jpeg',colors) 


    

        digits.updateGUI(color, depth)

    t.stop()


        
if __name__ == "__main__":
    main()
