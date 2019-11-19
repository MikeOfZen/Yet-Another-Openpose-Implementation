from config import *
import coco_helper as ch
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import tensorflow as tf

def show_by_id(id):
    """use to show a single image from the databse by its id"""
    display(Image(filename=ch.id_to_filename(id)))

def image_by_id(id,resize=True):
    """returns an array of the image"""
    f=tf.io.read_file(ch.id_to_filename(id))
    img= tf.image.decode_jpeg(f)
    img= tf.image.convert_image_dtype(img,tf.float32)
    if resize:
        img=tf.image.resize(img,IMAGE_SIZE)
    return img

def to_3_channels(one_channel,channel=0):
    def rotate(l, x):
        return l[-x:] + l[:-x]
    zeros=np.zeros_like(one_channel)
    channels=(one_channel,zeros,zeros)
    return np.stack(rotate(channels,channel),axis=-1)

def plot_vector_field(v_field, downsample=5):
    """if downsample is 1, original size is returned"""
    plt.figure(figsize=(8,8))
    if downsample:
        U= v_field[::downsample, ::downsample, 0]
        V= v_field[::downsample, ::downsample, 1]
    plt.quiver(U,V,scale=5,angles="xy")

def plot_PAFs_on_img(PAFs_array, img, downsample=3):
    """if downsample is 1, original size is returned
    :param PAFs_array must by np.ndarray"""
    plt.figure(figsize=(8,8))

    cmap = plt.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=0, vmax=PAFs_array.shape[0])

    for i,PAF in enumerate(PAFs_array):
        pruned_field=prune_quiver(PAF,downsample)

        U=pruned_field[...,0]
        V=pruned_field[...,1]

        plt.quiver(U,V,scale=20,angles="xy",minlength=0.1,linewidth=0.1,color=cmap(norm(i)))
    plt.imshow(img)
    plt.show()

def prune_quiver(v_field,downsample=5):
    canvas=np.zeros_like(v_field,dtype=np.bool)
    canvas[::downsample, ::downsample, :]=True
    canvas= np.invert( canvas)
    v_field[canvas]=np.zeros_like(v_field.shape[-1])
    return v_field


def plot_skeleton_on_img(PAFs_array, img):
    """if downsample is 1, original size is returned
    :param PAFs_array must be np.ndarray
    :param img must be ndarray"""
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=0, vmax=PAFs_array.shape[0])

    scalar_PAF=np.linalg.norm(PAFs_array,axis=-1)
    colored_PAF=scalar_PAF*np.arange(1, PAFs_array.shape[0]+1)[:, None, None]
    compressed_PAF=np.max(colored_PAF,axis=0)
    actual_colors=cmap(norm(compressed_PAF))
    actual_colors[compressed_PAF==0]=np.zeros(4)

    plt.imshow(img)
    plt.imshow(actual_colors)
    plt.show()