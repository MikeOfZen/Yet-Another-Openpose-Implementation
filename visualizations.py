#from config import *
import coco_helper as ch
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import tensorflow as tf

def show_by_id(id):
    """use to show a single image from the databse by its id"""
    display(Image(filename=ch.id_to_filename(id)))

# def image_by_id(id,resize=True):
#     """returns an array of the image"""
#     f=tf.io.read_file(ch.id_to_filename(id))
#     img= tf.image.decode_jpeg(f)
#     img= tf.image.convert_image_dtype(img,tf.float32)
#     if resize:
#         img=tf.image.resize(img,IMAGE_SIZE)
#     return img

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

# def plot_PAFs_on_img(PAFs_array, img, downsample=3):
#     """if downsample is 1, original size is returned
#     :param PAFs_array must by np.ndarray"""
#     plt.figure(figsize=(8,8))
#
#     cmap = plt.cm.hsv
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=PAFs_array.shape[0])
#
#     for i,PAF in enumerate(PAFs_array):
#         pruned_field=prune_quiver(PAF,downsample)
#
#         U=pruned_field[...,0]
#         V=pruned_field[...,1]
#
#         plt.quiver(U,V,scale=20,angles="xy",minlength=0.1,linewidth=0.1,color=cmap(norm(i)))
#     plt.imshow(img)
#     plt.show()

# def prune_quiver(v_field,downsample=5):
#     canvas=np.zeros_like(v_field,dtype=np.bool)
#     canvas[::downsample, ::downsample, :]=True
#     canvas= np.invert( canvas)
#     v_field[canvas]=np.zeros_like(v_field.shape[-1])
#     return v_field


# def plot_skeleton_on_img(PAFs_array, img):
#     """if downsample is 1, original size is returned
#     :param PAFs_array must be np.ndarray
#     :param img must be ndarray"""
#     plt.figure(figsize=(8, 8))
#     cmap = plt.cm.hsv
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=PAFs_array.shape[0])
#
#     scalar_PAF=np.linalg.norm(PAFs_array,axis=-1)
#     colored_PAF=scalar_PAF*np.arange(1, PAFs_array.shape[0]+1)[:, None, None]
#     compressed_PAF=np.max(colored_PAF,axis=0)
#     actual_colors=cmap(norm(compressed_PAF))
#     actual_colors[compressed_PAF==0]=np.zeros(4)
#
#     plt.imshow(img)
#     plt.imshow(actual_colors)
#     plt.show()

def show_img_pafs_kpts(img,pafs=None,kpts=None,squeeze_kpts=5,kpts_alpha=0.6,figure_size=8):
    """Draws an image, a keypoints layer, a part affinity field vector field, all three, or any combintaion thereof
    *the PAF array shape should be somewhat smaller ~x4 than the image to not overwhelm it.
    *doesnt work on batch
    :param pafs must by np.ndarray of the PAFs, shape =(1,h,w,num_joints*2) or (h,w,num_joints*2)
    :param kpts must by np.ndarray of the kpts, shape =(1,h,w,num_kpts) or (h,w,num_kpts)
    :param squeeze_kpts determines how 'squeezed' in space the kpts are, a higher number will make the kpts smaller
    either or are optional
    :param kpts_alpha float 0..1 range for the transperency intesity of the kpts
    """
    assert type(img) is np.ndarray or type(kpts) is np.ndarray or type(pafs) is np.ndarray , "Missing input or not numpy.ndarray"

    plt.figure(figsize=(figure_size,figure_size))

    kwargs={}
    if type(pafs) is np.ndarray:
        pafs = np.squeeze(pafs)  # from batch to single
        kwargs={"extent":(0,pafs.shape[1],pafs.shape[0],0)}
    if type(kpts) is np.ndarray:
        kpts = np.squeeze(kpts)  # from batch to single
        kwargs={"extent":(0,kpts.shape[1],kpts.shape[0],0)}

    if type(img) is np.ndarray:
        img = np.squeeze(img)
        plt.imshow(img,**kwargs)
    if type(kpts) is np.ndarray:
        draw_kpts(kpts,squeeze_kpts,kpts_alpha)
    if type(pafs) is np.ndarray:
        draw_pafs(pafs)
    plt.show()

def draw_pafs(pafs):
    cmap = plt.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=0, vmax=pafs.shape[0])

    num_pafs=int(pafs.shape[-1] / 2)
    for i in range(num_pafs):
        # pruned_field=prune_quiver(PAF,downsample)
        Y = pafs[..., i]
        X = pafs[..., num_pafs+i]
        plt.quiver(X, Y, scale=20, angles="xy", minlength=0.1, linewidth=0.1, color=cmap(norm(i)))

def draw_kpts(kpts,squeeze=1,kpts_alpha=0.6):
    superimposed_kpts=kpts.max(axis=-1)
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    spots = cmap(norm(superimposed_kpts))

    alpha=(superimposed_kpts)**squeeze / superimposed_kpts.max()
    alpha=alpha*kpts_alpha
    spots[..., 3] = alpha
    plt.imshow(spots)


def plot_fields(*fields, colorbars=False):
    num = len(fields)
    cols = np.ceil(np.sqrt(num))
    rows = np.ceil(num / cols)
    for i, field in enumerate(fields):
        plt.subplot(rows, cols, i+1)
        plt.imshow(field)
        if colorbars: plt.colorbar()
