import config as c
import os

def id_to_filename(id):
    return os.path.join(c.IMAGES_PATH,"{:012}.jpg".format(id))

