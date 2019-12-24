def get_checkpoints(config):
    if config.TPU_MODE:
        from google.cloud import storage

        def _get_checkpoints(config):
            checkpoints = []
            storage_client = storage.Client()
            for i, f in enumerate(storage_client.list_blobs(config.RESULTS_BUCKETNAME, prefix=config.CHECKPOINTS_DIR)):
                if f.name.endswith('.index'):
                    checkpoint_name = "gs://" + config.RESULTS_BUCKETNAME + "/" + f.name[:-6]
                    checkpoints.append(checkpoint_name)
            return checkpoints
    else:
        import glob

        def _get_checkpoints(config):
            checkpoints = glob.glob(config.CHECKPOINTS_PATH + "*.ckpt.index")
            checkpoints = [x[:-6] for x in checkpoints]
            return checkpoints
    return _get_checkpoints(config)


def checkpoints_prompt(config):
    checkpoints = get_checkpoints(config)
    if not checkpoints:
        print("Found no checkpoints")
        return None, 0
    print("Found these checkpoints:")
    print("0.Dont load checkpoint")
    for i, checkpoint in enumerate(checkpoints):
        print(i + 1, "." + checkpoint, flush=True)
    options = [str(x) for x in range(len(checkpoints) + 1)]
    selection = ""
    while selection not in options:
        selection = input("Please select checkpoint, or 0 to continue without loading:")
    selection = int(selection)
    checkpoint = checkpoints[selection - 1] if selection else None
    epoch = get_epoch_from_name(checkpoint) if selection else 0
    return checkpoint, epoch


def get_epoch_from_name(checkpoint_name):
    epoch_str = checkpoint_name[-9:-5]
    return int(epoch_str)
