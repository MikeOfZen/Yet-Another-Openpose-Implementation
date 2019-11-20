import config as c
import os

def id_to_filename(id):
    return os.path.join(c.IMAGES_PATH,f"{id:012}.jpg")

