import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage import color

hue_gradient = np.linspace(0, 1)
hsv = np.ones(shape=(1, len(hue_gradient), 3), dtype=float)
hsv[:, :, 0] = hue_gradient

all_hues = color.hsv2rgb(hsv)


def plot_hsv(image, bins=12):
    """
    Plot HSV histograms of image
    INPUT: image with HSV channels
    OUPUT: plot of HSV histograms and color spectrum
    """
    sns.set_style("whitegrid", {'axes.grid': False})
    fig = plt.figure(figsize=(12, 5))
    plt.subplots_adjust(top=2, bottom=1, wspace=.5, hspace=0)
    plt.subplot(231)
    plt.hist(image[:, :, 0].flatten(), bins=bins, color='gray')
    plt.title('Hue')
    plt.subplot(232)
    plt.hist(image[:, :, 1].flatten(), bins=bins, color='gray')
    plt.title('Saturation')
    plt.subplot(233)
    plt.hist(image[:, :, 2].flatten(), bins=bins, color='gray')
    plt.title('Value')
    plt.subplot(234)
    plt.imshow(all_hues, extent=(0, 1, 0, 0.2))
    plt.show()


def plot_kmeans(features, labels, centers, x, y):
    d = {0: 'r', 1: 'b', 2: 'g', 3: 'c', 4: 'm'}
    colors = []
    for i in labels:
        colors.append(d[i])
    # plot the kmeans clustering
    plt.figure(figsize=(10, 6))
    plt.xlim(features[:, x].min() - 0.1 * features[:, x].max(),
             features[:, x].max() + 0.1 * features[:, x].max())
    plt.ylim(features[:, y].min() - 0.1 * features[:, y].max(),
             features[:, y].max() + 0.1 * features[:, y].max())
    plt.scatter(features[:, x], features[:, y],
                c=colors, edgecolors='face', alpha=0.3)
    plt.scatter(center[:, x],
                centers[:, y],
                c=['r', 'b', 'g', 'c', 'm'], s=50)
    plt.show()
