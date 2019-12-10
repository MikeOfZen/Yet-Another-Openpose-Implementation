"""Run this script to verify proper connectivity and permissions are available"""

from sys import platform
import subprocess
from tpu_training.TPU_config import *

if platform not in ["linux","linux2"]:
    raise OSError("This script must be run from the GCS VM corresponding to the TPU")

def test_connectivity():
    
    result=subprocess.run(["touch","/tmp/test"])  
    print(result.stdout)
    print("Not OK" if result.returncode else "OK")
    
    print("\nChecking tensorboard output directory permissions")
    result=subprocess.run(["gsutil","cp /tmp/test %s/test"%TENSORBOARD_PATH])
    print(result.stdout)
    print("Not OK" if result.returncode else "OK")
    result=subprocess.run(["gsutil", "rm %s/test"%TENSORBOARD_PATH])
    print(result.stdout)
    print("Not OK" if result.returncode else "OK")
    
    print("\nChecking checkpoints output directory permissions")
    result=subprocess.run(["gsutil","cp /tmp/test %s/test"%CHECKPOINTS_PATH])
    print(result.stdout)
    print("Not OK" if result.returncode else "OK")
    result=subprocess.run(["gsutil","rm %s/test"%CHECKPOINTS_PATH])
    print(result.stdout)
    print("Not OK" if result.returncode else "OK")
    
    print("\nChecking TPU connectivity")
    result=subprocess.check_output(["nmap","-Pn -p8470 %s"%TPU_IP])
    print(str(result.stdout))
    print("Not OK" if result.returncode else "OK")