# -*- coding: utf-8 -*-


import sys
sys.path.append('../')

# libraries
import numpy as np
from scipy.ndimage import gaussian_filter


def plot_wam(ax,wam,levels,smooth=False, sigma=1, cmap="viridis", normalize_approx=False):
    """
    wrapper that plots the wam
    """

    size = wam.shape[0]
    display=np.zeros(wam.shape)

    if normalize_approx:
        boundaries=int(size / 2**levels)
        approx_coeffs=wam[:boundaries,:boundaries]

        display=wam/wam.max()
        display[:boundaries,:boundaries]=0


    else:
        display=wam

    if smooth:
        ax.imshow(gaussian_filter(display, sigma=sigma), cmap=cmap)
    else:
        ax.imshow(display, cmap=cmap)

    add_lines(size, levels, ax)


def plot_wavelet_regions(size,levels):
    """
    returns the dictonnaries with the
    coordinates of the lines for the plots
    """

    center = size // 2
    h, v = {}, {} # dictionnaries that will store the lines
    # initialize the first level
    h[0] = np.array([
        [0, center],
        [size,center],
    ])
    v[0] = np.array([
        [center,size],
        [center,0],
    ])
    # define the horizontal and vertical lines at each level
    for i in range(1, levels):
        h[i] = h[i-1] // 2
        h[i][:,1]
        v[i] = v[i-1] // 2
        v[i][:,1] 
        
    return h, v   

def add_lines(size, levels, ax):
    """
    add white lines to the ax where the 
    wam is plotted
    """
    h, v = plot_wavelet_regions(size, levels)

    ax.set_xlim(0,size)
    ax.set_ylim(size,0)

    for k in range(levels):
        ax.plot(h[k][:,0], h[k][:,1], c = 'w')
        ax.plot(v[k][:,0], v[k][:,1], c = 'w')

    return None

