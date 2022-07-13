# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from numpy import concatenate, mean, size, float64, float32, uint8
import cv2
import matplotlib.pyplot as plt
import hydra
import pybullet as p
import pybulletX as px
import tacto  # Import TACTO
from math import floor

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
        colors = concatenate(color, axis=1)        
        depths =concatenate(list(map(digits._depth_to_color, depth)), axis=1)  
                 
        #img=cv2.cvtColor(depths,cv2.COLOR_BGR2RGB)
        vectorized = colors.reshape((-1,3))
        vectorized = float32(vectorized)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        attempts=20
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((colors.shape))
        figure_size = 15
        plt.figure(figsize=(figure_size,figure_size))
        plt.subplot(1,2,1),plt.imshow(colors)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(result_image)
        plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
        plt.show()
        
        #depths = Compute_BW_Contact_Intensity(depths)           
        #depths = cv2.resize(depths, (500,500), interpolation = cv2.INTER_AREA)

        cv2.imshow("depths", cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))
        #cv2.imshow("colors", cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))


        digits.updateGUI(color, depth)

    t.stop()

def Compute_BW_Contact_Intensity(depths):
    intense_contact =  [(i,j)  for i, rows in enumerate(depths) for j,pixel in enumerate(rows) if (pixel[0] >= 200)]
    medium_contact = [(i,j)  for i, rows in enumerate(depths) for j,pixel in enumerate(rows) if (pixel[0] >= 100 and pixel[0] < 200)]
    weak_contact = [(i,j)  for i, rows in enumerate(depths) for j,pixel in enumerate(rows) if (pixel[0] >= 50 and  pixel[0] < 100)]
        
        #print("\n\nNumber of intense pixels: ", len(intense_contact), "\nNumber of medium pixels: ", len(medium_contact), "\nNumber of weak pixels: ", len(weak_contact))
    if (intense_contact and medium_contact and weak_contact):
        intense_contact_center = mean(intense_contact, axis=0, dtype=float64)
        medium_contact_center = mean(medium_contact, axis=0, dtype=float64)
        weak_contact_center = mean(weak_contact, axis=0, dtype=float64)

        print(intense_contact_center, medium_contact_center, weak_contact_center)

        depths = cv2.circle(depths, (floor(intense_contact_center[1]),floor(intense_contact_center[0])), 0, [255, 0, 0], 2)
        depths = cv2.circle(depths, (floor(medium_contact_center[1]),floor(medium_contact_center[0])), 0, [200, 0, 200], 2)
        depths = cv2.circle(depths, (floor(weak_contact_center[1]),floor(weak_contact_center[0])), 0, [0,0,2500], 2)
    return depths


        
if __name__ == "__main__":
    main()
