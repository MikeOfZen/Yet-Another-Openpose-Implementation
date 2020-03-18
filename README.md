Yet Another Openpose Implementation
---
The Openpose algorithm processes an image with a deep CNN and extracts a skeleton representation of the pose of multiple people from it

This project reimplemented from scratch the [OpenPose paper](https://arxiv.org/abs/1812.08008) [1], Using Tensorflow 2.1 
and TPU powered training (optional).

Sample result:<br>
![Video result](doc/YAOP_gif_smallres.gif)

The default dataset used for training is the [COCO 2017 keypoints dataset](http://cocodataset.org/) 

### Demo applications usage
The repo contains a trained model, and the applications are for demonstration purposes.
* Install all dependencies.
* For the web-cam demo launch `applications/cam.py`
* For video annotation launch `applications/video.py [input_video_filepath] [output_video_filepath] --fourcc [installed codec fourcc, for example XVID] --fps [the input video frame rate]` 

---
### Training setup
It's possible to train locally with a strong gpu (an epoch time of a few hours) in which case no TPU setup is required, or setup a google cloud
tpu account and train on a TPU (an epoch time of ~15min, on a v2-8 instance).
##### TPU setup
1. Set a google cloud account
2. Create TPU instance (version must be 2.x+(for now it's a nighlty release))
3. Create storage bucket for the TFrecords and for the training results in the same zone (must be globally unique names).
4. Create a VM control instance (ie, a regular VM to utilize the TPU) in the same zone as the TPU. <br>
   *a suitable image is [`tf2-latest-cpu`](https://cloud.google.com/ai-platform/deep-learning-vm/docs/images)
5. The network connectivity and permissions should allow full access between all 3 (TPU,VM,Bucket).<br>
   *All VM scopes should be allowed <br>
   *The bucket should be accessible with full permissions from the TPU and the VM  
6. On the VM install `gcsfuse` (`dataset/install_gcsfuse.sh`)
7. Setup Jupyter on the control VM to be accessible remotely

#### Training 
If training locally, this applies to the local machine, if using TPU, to the control VM.
##### local Training
1. Checkout this repo<br> 
2. Install dependencies <br>
*if able to use tensorflow-gpu, install it <br> 
3. Download the dataset using `dataset/get_data` (.sh or .bat) (run from its own working dir)
4. Run `training/transform_dataset.py` (from it's wdir), this creates the TFrecord files used in training<br>
7. Update `configs/local_storage_config`
5. Run `training/Train.ipynb` using Jupyter
9. Run and access Tensorboard to track the training progress
10. When happy with the results, update the demo apps with the new model.
 
##### TPU Training
1. Checkout this repo<br> 
2. Install dependencies <br>
3. Download the dataset using `dataset/get_data.sh`(run from its own working dir)
4. Mount the GCS bucket using `dataset/mount_bucket.sh`
5. Run `training/transform_dataset.py` (from it's wdir), this creates the TFrecord files used in training<br>
6. From within GCS verify the TFrecord files are in place
7. Update `configs/remote_storage_config`
7. Open `training/Train.ipynb` from Jupyter
8. Check `Train.ipynb` settings and run the training.
9. Run and access Tensorboard to track the training progress
10. When happy with the results, copy over the trained model to the local machine, and update the demo apps with the new model.
---

##### Dependencies

`Python 3.5+` And everything in `requirements.txt` <br>
*`pycocotools` for windows can be installed by:
`pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

---
### References
[1]  Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh, Openpose: Realtime
multi-person 2d pose estimation using part affinity fields, 2018.
