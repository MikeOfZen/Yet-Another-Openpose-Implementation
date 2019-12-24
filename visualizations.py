import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors


def to_3_channels(one_channel, channel=0):
    def rotate(l, x):
        return l[-x:] + l[:-x]

    zeros = np.zeros_like(one_channel)
    channels = (one_channel, zeros, zeros)
    return np.stack(rotate(channels, channel), axis=-1)


def plot_vector_field(v_field, downsample=1):
    """if downsample is 1, original size is returned"""
    plt.figure(figsize=(8, 8))
    U = v_field[::downsample, ::downsample, 0]
    V = v_field[::downsample, ::downsample, 1]
    plt.quiver(U, V, scale=5, angles="xy")


def show_img_pafs_kpts(img, pafs=None, kpts=None, mask=None, squeeze_kpts=5, kpts_alpha=0.6, figure_size=8):
    """Draws an image, a keypoints layer, a part affinity field vector field, all three, or any combination thereof
    *the PAF array shape should be somewhat smaller ~x4 than the image to not overwhelm it.
    *doesnt work on batch
    :param figure_size: size in inches of the figure
    :param img: must by np.ndarray of the image
    :param pafs must by np.ndarray of the PAFs, shape =(1,h,w,num_joints*2) or (h,w,num_joints*2)
    :param kpts must by np.ndarray of the kpts, shape =(1,h,w,num_kpts) or (h,w,num_kpts)
    :param mask must by np.ndarray of the mask, shape =(1,h,w,1) or (h,w,1)
    :param squeeze_kpts determines how 'squeezed' in space the kpts are, a higher number will make the kpts smaller
    either or are optional
    :param kpts_alpha float 0..1 range for the transparency intensity of the kpts
    """
    assert type(img) is np.ndarray or type(kpts) is np.ndarray or type(pafs) is np.ndarray, "Missing input or not numpy.ndarray"

    plt.figure(figsize=(figure_size, figure_size))

    kwargs = {}
    if type(pafs) is np.ndarray:
        try:
            pafs = np.squeeze(pafs, axis=0)  # from batch to single
        except ValueError:
            pass
        kwargs = {"extent": (0, pafs.shape[1] - 1, pafs.shape[0] - 1, 0)}
    if type(kpts) is np.ndarray:
        try:
            kpts = np.squeeze(kpts, axis=0)  # from batch to single
        except ValueError:
            pass
        kwargs = {"extent": (0, kpts.shape[1] - 1, kpts.shape[0] - 1, 0)}

    if type(img) is np.ndarray:
        img = np.squeeze(img)
        plt.imshow(img, **kwargs)
    if type(mask) is np.ndarray:
        mask = np.squeeze(mask)
        plt.imshow(mask, alpha=0.3, cmap='gray', **kwargs)
    if type(kpts) is np.ndarray:
        draw_kpts(kpts, squeeze_kpts, kpts_alpha)
    if type(pafs) is np.ndarray:
        draw_pafs(pafs)
    plt.show()


def draw_pafs(pafs):
    cmap = plt.cm.hsv
    norm = matplotlib.colors.Normalize(vmin=0, vmax=pafs.shape[0])

    num_pafs = int(pafs.shape[-1] / 2)
    for i in range(num_pafs):
        # pruned_field=prune_quiver(PAF,downsample)
        Y = pafs[..., i]
        X = pafs[..., num_pafs + i]
        plt.quiver(X, Y, scale=20, angles="xy", minlength=0.1, linewidth=0.1, color=cmap(norm(i)))


def draw_kpts(kpts, squeeze=1, kpts_alpha=0.6):
    superimposed_kpts = kpts.max(axis=-1)
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    spots = cmap(norm(superimposed_kpts))

    alpha = superimposed_kpts ** squeeze / superimposed_kpts.max()
    alpha = alpha * kpts_alpha
    spots[..., 3] = alpha
    plt.imshow(spots)


def plot_fields(*fields, colorbars=False):
    num = len(fields)
    cols = np.ceil(np.sqrt(num))
    rows = np.ceil(num / cols)
    for i, field in enumerate(fields):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(field)
        if colorbars: plt.colorbar()
