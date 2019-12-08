DS_NUM_KEYPOINTS=17

#our newly defined ketpoints
KEYPOINTS_DEF=(
    {"name":'nose'         ,"ds_idxs":0            },  #0
    {"name":'neck'         ,"ds_idxs":(5,6)        },  #1
    {"name":'Rshoulder'    ,"ds_idxs":6            },  #2
    {"name":'Relbow'       ,"ds_idxs":8            },  #3
    {"name":'Rwrist'       ,"ds_idxs":10           },  #4
    {"name":'Lshoulder'    ,"ds_idxs":5            },  #5
    {"name":'Lelbow'       ,"ds_idxs":7            },  #6
    {"name":'Lwrist'       ,"ds_idxs":9           },  #7
    {"name":'Rhip'         ,"ds_idxs":12           },  #8
    {"name":'Rknee'        ,"ds_idxs":14           },  #9
    {"name":'Rankle'       ,"ds_idxs":16           },  #10
    {"name":'Lhip'         ,"ds_idxs":11           },  #11
    {"name":'Lknee'        ,"ds_idxs":13           },  #12
    {"name":'Lankle'       ,"ds_idxs":15           },  #13
    {"name":'Reye'         ,"ds_idxs":2            },  #14
    {"name":'Leye'         ,"ds_idxs":1            },  #15
    {"name":'Rear'         ,"ds_idxs":4            },  #16
    {"name":'Lear'         ,"ds_idxs":3            },  #17
)

#our newly defined joints (disregarding coco defined ones)
JOINTS_DEF=(
    (0,1),      #0 nose to neck
    (1,2),      #1
    (2,3),      #2
    (3,4),      #3
    (1,5),      #4
    (5,6),      #5
    (6,7),      #6
    (1,8),      #7
    (8,9),      #8
    (9,10),     #10
    (1,11),     #11
    (11,12),    #12
    (12,13),    #13
    (0,14),     #14
    (14,16),    #15
    (0,15),     #16
    (15,17),    #17
)

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
