"""Run this script to verify proper connectivity and permissions are available"""

from sys import platform
import os
from tpu_training.config_tpu import *

if platform not in ["linux","linux2"]:
    raise OSError("This script must be run from the GCS VM corresponding to the TPU")

os.system("touch /tmp/test")
print("\nChecking tensorboard output directory permissions")
os.system("gsutil cp /tmp/test %s/test"%TENSORBOARD_PATH)
os.system("gsutil rm %s/test"%TENSORBOARD_PATH)
print("\nChecking checkpoints output directory permissions")
os.system("gsutil cp /tmp/test %s/test"%CHECKPOINTS_PATH)
os.system("gsutil rm %s/test"%CHECKPOINTS_PATH)

print("\nChecking TPU connectivity")
os.system("nmap -Pn -p8470 %s"%TPU_IP)