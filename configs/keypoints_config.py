import collections

DS_NUM_KEYPOINTS=17

# our newly defined keypoints
#for mirror augmentaiton to work, the order of keypoitns,and joints must be center,right,left
KEYPOINTS_DEF={
 'nose'         :{"idx":0   ,"name":'nose'         ,"side":"C"   ,"ds_idxs":0    ,"mirror_name":None       },
 'sternum'      :{"idx":1   ,"name":'sternum'      ,"side":"C"   ,"ds_idxs":(5,6),"mirror_name":None       },
 'Rshoulder'    :{"idx":2   ,"name":'Rshoulder'    ,"side":"R"   ,"ds_idxs":6    ,"mirror_name":'Lshoulder'},
 'Relbow'       :{"idx":3   ,"name":'Relbow'       ,"side":"R"   ,"ds_idxs":8    ,"mirror_name":'Lelbow'   },
 'Rwrist'       :{"idx":4   ,"name":'Rwrist'       ,"side":"R"   ,"ds_idxs":10   ,"mirror_name":'Lwrist'   },
 'Rhip'         :{"idx":5   ,"name":'Rhip'         ,"side":"R"   ,"ds_idxs":12   ,"mirror_name":'Lhip'     },
 'Rknee'        :{"idx":6   ,"name":'Rknee'        ,"side":"R"   ,"ds_idxs":14   ,"mirror_name":'Lknee'    },
 'Rankle'       :{"idx":7   ,"name":'Rankle'       ,"side":"R"   ,"ds_idxs":16   ,"mirror_name":'Lankle'   },
 'Reye'         :{"idx":8   ,"name":'Reye'         ,"side":"R"   ,"ds_idxs":2    ,"mirror_name":'Leye'     },
 'Rear'         :{"idx":9   ,"name":'Rear'         ,"side":"R"   ,"ds_idxs":4    ,"mirror_name":'Lear'     },
 'Lshoulder'    :{"idx":10  ,"name":'Lshoulder'    ,"side":"L"   ,"ds_idxs":5    ,"mirror_name":'Rshoulder'},
 'Lelbow'       :{"idx":11  ,"name":'Lelbow'       ,"side":"L"   ,"ds_idxs":7    ,"mirror_name":'Relbow'   },
 'Lwrist'       :{"idx":12  ,"name":'Lwrist'       ,"side":"L"   ,"ds_idxs":9    ,"mirror_name":'Rwrist'   },
 'Lhip'         :{"idx":13  ,"name":'Lhip'         ,"side":"L"   ,"ds_idxs":11   ,"mirror_name":'Rhip'     },
 'Lknee'        :{"idx":14  ,"name":'Lknee'        ,"side":"L"   ,"ds_idxs":13   ,"mirror_name":'Rknee'    },
 'Lankle'       :{"idx":15  ,"name":'Lankle'       ,"side":"L"   ,"ds_idxs":15   ,"mirror_name":'Rankle'   },
 'Leye'         :{"idx":16  ,"name":'Leye'         ,"side":"L"   ,"ds_idxs":1    ,"mirror_name":'Reye'     },
 'Lear'         :{"idx":17  ,"name":'Lear'         ,"side":"L"   ,"ds_idxs":3    ,"mirror_name":'Rear'     }
        }
KEYPOINTS_DEF=collections.OrderedDict(sorted(KEYPOINTS_DEF.items(), key=lambda t: t[1]["idx"]))
KEYPOINTS_SIDES={"C":(0,1),"R":(2,9),"L":(10,17)} #the starting and ending indexes for the center,right and left sides from KEYPOINTS_DEF above

#our newly defined joints (disregarding coco defined ones)
JOINTS_DEF={
"neck"          :{"idx":0  ,"kpts":('nose','sternum'),        "side":"C",  "name":"neck"      ,"other_side_idx":None       },
"Rshoulder"     :{"idx":1  ,"kpts":('sternum','Rshoulder'),   "side":"R",  "name":"Rshoulder" ,"other_side_idx":"Lshoulder"},
"RupperArm"     :{"idx":2  ,"kpts":('Rshoulder','Relbow'),    "side":"R",  "name":"RupperArm" ,"other_side_idx":"LupperArm"},
"Rlowerarm"     :{"idx":3  ,"kpts":('Relbow','Rwrist'),       "side":"R",  "name":"Rlowerarm" ,"other_side_idx":"Llowerarm"},
"Rbodyside"     :{"idx":4  ,"kpts":('sternum','Rhip'),        "side":"R",  "name":"Rbodyside" ,"other_side_idx":"Lbodyside"},
"Rupperleg"     :{"idx":5  ,"kpts":('Rhip','Rknee'),          "side":"R",  "name":"Rupperleg" ,"other_side_idx":"Lupperleg"},
"Rlowerleg"     :{"idx":6  ,"kpts":('Rknee','Rankle'),        "side":"R",  "name":"Rlowerleg" ,"other_side_idx":"Llowerleg"},
"Rchick"        :{"idx":7  ,"kpts":('nose','Reye'),           "side":"R",  "name":"Rchick"    ,"other_side_idx":"Lchick"   },
"Rtemple"       :{"idx":8  ,"kpts":('Reye','Rear'),           "side":"R",  "name":"Rtemple"   ,"other_side_idx":"Ltemple"  },
"Lshoulder"     :{"idx":9  ,"kpts":('sternum','Lshoulder'),   "side":"L",  "name":"Lshoulder" ,"other_side_idx":"Rshoulder"},
"LupperArm"     :{"idx":10 ,"kpts":('Lshoulder','Lelbow'),    "side":"L",  "name":"LupperArm" ,"other_side_idx":"RupperArm"},
"Llowerarm"     :{"idx":11 ,"kpts":('Lelbow','Lwrist'),       "side":"L",  "name":"Llowerarm" ,"other_side_idx":"Rlowerarm"},
"Lbodyside"     :{"idx":12 ,"kpts":('sternum','Lhip'),        "side":"L",  "name":"Lbodyside" ,"other_side_idx":"Rbodyside"},
"Lupperleg"     :{"idx":13 ,"kpts":('Lhip','Lknee'),          "side":"L",  "name":"Lupperleg" ,"other_side_idx":"Rupperleg"},
"Llowerleg"     :{"idx":14 ,"kpts":('Lknee','Lankle'),        "side":"L",  "name":"Llowerleg" ,"other_side_idx":"Rlowerleg"},
"Lchick"        :{"idx":15 ,"kpts":('nose','Leye'),           "side":"L",  "name":"Lchick"    ,"other_side_idx":"Rchick"   },
"Ltemple"       :{"idx":16 ,"kpts":('Leye','Lear'),           "side":"L",  "name":"Ltemple"   ,"other_side_idx":"Rtemple"  }
}
JOINTS_DEF=collections.OrderedDict(sorted(JOINTS_DEF.items(), key=lambda t: t[1]["idx"]))
JOINTS_SIDES={"C":(0,0),"R":(1,8),"L":(9,16)} #the starting and ending indexes for the center,right and left sides from JOINTS_DEF above


#taken directly from the annotations JSON file
#COCO_DATASET_KPTS=['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
#                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
#                'right_knee', 'left_ankle', 'right_ankle']

#shifted by -1 to match keypoints idx
#COCO_DATASET_JOINTS=[[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
#                [ 1, 2], [ 0, 1], [ 0, 2], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6]]
#

# #mapping from coco kpts position to name
# map_ds_kpts={
#      0 :   'nose'
#     ,1 :   'Leye'
#     ,2 :   'Reye'
#     ,3 :   'Lear'
#     ,4 :   'Rear'
#     ,5 :   'Lshoulder'
#     ,6 :   'Rshoulder'
#     ,7 :   'Lelbow'
#     ,8 :   'Relbow'
#     ,9:   'Lwrist'
#     ,10:   'Rwrist'
#     ,11:   'Lhip'
#     ,12:   'Rhip'
#     ,13:   'Lknee'
#     ,14:   'Rknee'
#     ,15:   'Lankle'
#     ,16:   'Rankle'
# }
