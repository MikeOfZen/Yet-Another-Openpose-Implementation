from config import *


if TPU_MODE:
    from google.cloud import storage
    def get_checkpoints():
        checkpoints=[]
        storage_client = storage.Client()
        for i,f in enumerate( storage_client.list_blobs(RESULTS_BUCKETNAME,prefix=CHECKPOINTS_DIR)):
            if f.name.endswith('.index'):
                checkpoint_name="gs://"+RESULTS_BUCKETNAME+"/"+f.name[:-6]
                checkpoints.append(checkpoint_name)
        return checkpoints
else:
    import glob
    def get_checkpoints():
        checkpoints = glob.glob(CHECKPOINTS_PATH + "*.ckpt.index")
        checkpoints=[x[:-6] for x in checkpoints]
        return checkpoints

def checkpoints_prompt():
    checkpoints = get_checkpoints()
    print("Found these checkpoints")
    print("0.Dont load checkpoint")
    for i, checkpoint in enumerate(checkpoints):
        print(i + 1, "." + checkpoint)
    options = [str(x) for x in range(len(checkpoints) + 1)]
    selection = ""
    while selection not in options:
        selection = input("Please select checkpoint, or 0 to continue without loading")
    selection = int(selection)
    checkpoint = checkpoints[selection - 1] if selection else None
    return checkpoint,get_epoch_from_name(checkpoint)

def get_epoch_from_name(checkpoint_name):
    epoch_str=checkpoint_name[-9:-5]
    return int(epoch_str)