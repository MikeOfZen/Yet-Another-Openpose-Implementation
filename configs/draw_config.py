import matplotlib.pyplot as plt
import matplotlib.colors

import configs.keypoints_config as kc

joint_line_thickness = 5
cmap = plt.cm.viridis
joints_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(kc.JOINTS_DEF))


def cmap_to_bgr(color):
    color = map(lambda x: x * 255, color)
    color = map(int, color)
    color = list(color)
    color = color[0:3]
    r = color[0]
    g = color[1]
    b = color[2]
    return b, g, r


joint_colors_bgr = {k: cmap_to_bgr(cmap(joints_norm(v["idx"]))) for k, v in kc.JOINTS_DEF.items()}

keypoint_circle_diameter = 10

DRAW_KEYPOINT_TEXT = True
keypoint_circle_color = (255, 255, 255)
keypoint_text_color = (125, 125, 125)
keypoint_text_thickness = 1
keypoint_text_scale = 0.5
