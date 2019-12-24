import json
import os
import objectpath
from itertools import groupby
import numpy as np

DATASET_PATH = r"./dataset/"
IMAGES_PATH = DATASET_PATH + "/train2017"
ANNOTATIONS_PATH = DATASET_PATH + "/annotations/person_keypoints_train2017.json"

with open(ANNOTATIONS_PATH, 'r') as f:
    json_file = json.load(f)
json_tree = objectpath.Tree(json_file)

img_keypts = [(x['image_id'], x['keypoints']) for x in json_tree.execute("$.annotations[@.num_keypoints is not 0]")]

del json_tree
del json_file

import gc

gc.collect()

print("Reading done, transforming now")
img_keypts = sorted(img_keypts, key=lambda x: x[0])
img_keypts = [list(g) for k, g in groupby(img_keypts, key=lambda x: x[0])]
img_keypts = [(x[0][0], [y[1] for y in x]) for x in img_keypts]
img_keypts = [(x[0], np.array(x[1], dtype=np.float32).reshape((-1, 17, 3))) for x in img_keypts]
print("before")
print(img_keypts[100])
