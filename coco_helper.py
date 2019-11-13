from IPython.display import Image, display

def id_to_filename(id):
    return os.path.join(IMAGES_PATH,f"{id:012}.jpg")

def show_by_id(id):
    display(Image(filename=id_to_filename(id)))
