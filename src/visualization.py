import cv2
import numpy as np
from env import GridWorld

def plot_heatmap(hm, obstacle, agetns):
    '''
    Inputs:
    hm: 2-D matrix with size (w, h). Heatmap.
    obstacle: Binary 2-D matrix with size (w, h).
    agents: A list contains all the agents of Agent type. Each one has 2-D position

    return:
    hm_show: RGB map containing heatmap, obstacles and agents.
    '''

    hm = cv2.applyColorMap(int(hm * 255), cv2.COLORMAP_JET)
    full_heatmap = cv2.resize(full_heatmap,(700,700),interpolation = cv2.INTER_AREA)
    cv2.imshow("heatMap", full_heatmap)