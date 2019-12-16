import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



y_grid = tf.linspace(0.0, 1.0, 100)
x_grid = tf.linspace(0.0, 1.0, 100)
yy, xx = tf.meshgrid(y_grid, x_grid, indexing='ij')  # indexing is a must, otherwise, it's just bizzare!
grid = tf.stack((yy, xx), axis=-1)


def keypoints_layer(kpts_layer):
    """This transforms a single layer of keypoints (such as 3 keypoints of type 'right shoulder')
    the keypoint_distance creates an array of the distances from each keypoint
    and this reduces them to a single array by the  of the distances.
    :param kpts_layer must be a tf.Tensor of shape (n,3)"""
    layer_dists = tf.map_fn(keypoint_distance, kpts_layer)
    all_dists=tf.math.reduce_min(layer_dists, axis=0)
    raw = tf.exp((-(all_dists ** 2) / 0.1))
    return raw

def keypoint_distance(kpt):
    """This transforms a single keypoint into an array of the distances from the keypoint
    :param kpt must be tf.Tensor of shape (x,y,a) where a is either 0,1,2 for missing,invisible and visible"""
    if kpt[2] == tf.constant(0.0):
        return tf.ones((100, 100), dtype=tf.float32)  # maximum distance incase of empty kpt, not ideal but meh
    else:
        ortho_dist = grid - kpt[0:2]
        return tf.linalg.norm(ortho_dist, axis=-1)

kpts=np.array([
        [0.3,0.3,2],
        [0.7,0.7,2],
        ],dtype=np.float32)
sample=keypoints_layer(kpts)

input = sample > 0.5

island = np.zeros((input.shape), dtype=np.int)
results = []
island_num = 1
head={}

for x in range(1, 46):
    for y in range(1, 46):
        if input[y, x]:
            above = island[y - 1, x]
            left = island[y, x - 1]
            if not above and not left:
                island[y , x]=island_num
                island_num += 1
            elif above and not left:
                island[y, x] = above
            elif not above and left:
                island[y, x] = left
            elif above and left:
                #make above child of left
                island[y, x] = left
                if above != left:
                    head[above]=left

plt.imshow(island)
plt.colorbar()
