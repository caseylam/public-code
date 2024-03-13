import numpy as np
from scipy.interpolate import splev, splrep
from alexmods.specutils.spectrum import read_mike_spectrum
import matplotlib.pyplot as plt
import copy
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.signal import argrelextrema

def median_data(file_names):
    """
    file_names : list of files to median
    """
    # should add special handling if only 2 images
    _data = fits.open(file_names[0])[0].data
    _median_stacked_data = np.zeros(_data.shape + tuple([len(file_names)]))
    for nn, file_name in enumerate(file_names):
        data = fits.open(file_name)[0].data
        _median_stacked_data[:,:,nn] = data
    med_data = len(file_names) * np.median(_median_stacked_data, axis=2)

    return med_data

def get_continuum(wl_in, fl_in, pstep=100, k=3, exclude_wl=None):
    """
    Based on https://github.com/howardisaacson/APF-BL-DAP/blob/main/Zoe/APFTutorial/APFtutorial.ipynb
    I think this is a somewhat simpler implementation.

    other choice, k=5
    
    wl_in : wavelength input
    fl_in : flux input
    pstep : size of bin in pixels
    """
    wl = copy.deepcopy(wl_in)
    fl = copy.deepcopy(fl_in)

    bins = np.arange(len(wl), step=pstep)
    # bins = np.linspace(0, len(wl)-1, pstep, dtype = int, endpoint=True)
    bins_wl = wl[bins]
    blaze_wl = 0.5 * (bins_wl[:-1] + bins_wl[1:])
    blaze_fl = np.zeros(len(blaze_wl))
    for ii in np.arange(len(blaze_wl)):
        idx = np.arange(bins[ii], bins[ii+1])
        # Figure out a better way to remove cosmic rays and spikes,
        # which tends to mess up things. Also find a way to mask absorption?
        diff = np.concatenate([np.array([0]), np.diff(fl[idx])])
        median, std = np.median(diff), np.std(diff)
        _keep_idx = np.where((diff > median - 3*std) &
                            (diff < median + 3*std))[0]
        _keep_idx2 = _keep_idx+1
        keep_idx = np.concatenate([_keep_idx, _keep_idx2])
        keep_idx = keep_idx[keep_idx < pstep]

        # Get the knots for the spline fit
        flux95 = np.quantile(fl[idx][keep_idx], 0.95)
        blaze_fl[ii] = flux95

    spl = splrep(blaze_wl, blaze_fl, s=500000, k=k)
    # This is the continuum
    blaze_fit = splev(wl, spl)

    # I'm not 100% convinced this step is needed and that it always does what I want.
    # first_normalized_flux = fl / blaze_fit
    # flux98 = np.quantile(first_normalized_flux, 0.98)
    # blaze_fit2 = blaze_fit / flux98

    return blaze_fit, blaze_wl, blaze_fl

def get_continuum_iterative(wl_in, fl_in, pstep=100, k=3, niters=0):
    """
    Based on https://github.com/howardisaacson/APF-BL-DAP/blob/main/Zoe/APFTutorial/APFtutorial.ipynb
    I think this is a somewhat simpler implementation.

    other choice, k=5
    
    wl_in : wavelength input
    fl_in : flux input
    pstep : size of bin in pixels
    """
    wl = copy.deepcopy(wl_in)
    fl = copy.deepcopy(fl_in)

    blaze_fit, blaze_wl, blaze_fl = get_continuum(wl_in, fl_in, pstep=100, k=3)
    
    for n in np.arange(niters):
        bins = np.arange(len(wl), step=pstep)
        # bins = np.linspace(0, len(wl)-1, pstep, dtype = int, endpoint=True)
        bins_wl = wl[bins]
        blaze_wl = 0.5 * (bins_wl[:-1] + bins_wl[1:])
        blaze_fl = np.zeros(len(blaze_wl))

        diff = blaze_fit - fl
        filtered = sigma_clip(diff)
        print(filtered.mask.sum())
        
        for ii in np.arange(len(blaze_wl)):
            idx = np.arange(bins[ii], bins[ii+1])
            # Get the knots for the spline fit
            fl_filt = copy.deepcopy(fl)
            fl_filt[filtered.mask] = np.nan
            flux95 = np.nanquantile(fl_filt[idx], 0.95)
            blaze_fl[ii] = flux95    
        spl = splrep(blaze_wl, blaze_fl, s=500000, k=k)
        # This is the continuum
        blaze_fit = splev(wl, spl)
        
    # I'm not 100% convinced this step is needed and that it always does what I want.
    # first_normalized_flux = fl / blaze_fit
    # flux98 = np.quantile(first_normalized_flux, 0.98)
    # blaze_fit2 = blaze_fit / flux98

    return blaze_fit, blaze_wl, blaze_fl

def split_mike_to_wl_fl_flerr(file_name):
    data = read_mike_spectrum(file_name)
    wl = np.zeros([len(data), 2048])
    fl = np.zeros([len(data), 2048])
    fl_err = np.zeros([len(data), 2048])
    for kk, key in enumerate(data.keys()):
        wl[kk] = data[key].dispersion
        fl[kk] = data[key].flux
        fl_err[kk] = 1/np.sqrt(data[key].ivar)

    return wl, fl, fl_err

def plot_raw_cont_norm(wl, fl, pstep, k):
    fig, ax = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    ymax = 0
    for ii in np.arange(wl.shape[0]):
        continuum, xknot, yknot = get_continuum(wl[ii], fl[ii], pstep=pstep, k=k)
        ax[0].plot(wl[ii][:-1], fl[ii][:-1])
        ax[0].plot(wl[ii][:-1], continuum[:-1], color='k', lw=0.5)
        ax[0].plot(xknot, yknot, '.', color='k')
        ax[1].plot(wl[ii][:-1], fl[ii][:-1]/continuum[:-1])
        ymax = np.max([ymax, np.quantile(fl[ii][:-1], 0.99)])
        ax[0].set_ylim(0, ymax)
        ax[1].set_ylim(0.05, 1.05)
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    ymax = 0
    for ii in np.arange(wl.shape[0]):
        continuum, xknot, yknot = get_continuum(wl[ii], fl[ii], pstep=pstep, k=k)
        ax[0].plot(wl[ii][:-1], fl[ii][:-1])
        ax[0].plot(wl[ii][:-1], continuum[:-1], color='k', lw=0.5)
        ax[0].plot(xknot, yknot, '.', color='k')
        # d_num = np.diff(continuum)[1:] - 2 * np.diff(continuum)[1:] + np.diff(continuum)[:-1]
        # d_den = np.diff(wl[ii])[1:]**2
        # ax[1].plot(wl[ii][1:-1], d_num/d_den)
        first_deriv = np.diff(continuum)/np.diff(wl[ii])
        first_deriv = np.concatenate([np.array([first_deriv[0]]), first_deriv])
        ax[1].plot(wl[ii][1:], np.diff(continuum)/np.diff(wl[ii]))
        idx_up = argrelextrema(first_deriv, np.greater)[0]
        idx_lo = argrelextrema(first_deriv, np.less)[0]
        ax[1].plot(wl[ii][1:][idx_up], (np.diff(continuum)/np.diff(wl[ii]))[idx_up], '.')
        ax[1].plot(wl[ii][1:][idx_lo], (np.diff(continuum)/np.diff(wl[ii]))[idx_lo], '.')
        # Figure out where the relative extrema have the biggest difference,
        # and then get rid of those spline points.
        idx_extrema = np.sort(np.concatenate([idx_up, idx_lo]))
        if len(idx_extrema) > 2:
            extrema_diff = np.diff(continuum[idx_extrema])
            extrema_diff_max_idx = np.abs(extrema_diff).argmax() + 1
            ax[1].plot(wl[ii][1:][idx_extrema[extrema_diff_max_idx]],
                       (np.diff(continuum)/np.diff(wl[ii]))[idx_extrema[extrema_diff_max_idx]], 'x')
            ax[1].plot(wl[ii][1:][idx_extrema[extrema_diff_max_idx + 1]],
                       (np.diff(continuum)/np.diff(wl[ii]))[idx_extrema[extrema_diff_max_idx + 1]], 'x')
        
        ymax = np.max([ymax, np.quantile(fl[ii][:-1], 0.99)])
        ax[0].set_ylim(0, ymax)
    plt.show()

def plot_raw_cont_norm_everything(wl, fl, pstep):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ymax = 0
    for ii in np.arange(wl.shape[0]):
        continuum, xknot, yknot = get_continuum_iterative(wl[ii], fl[ii], pstep=pstep, k=5, niters=0)
        # ax[0].plot(wl[ii][:-1], fl[ii][:-1])
        ax[0].plot(wl[ii][:-1], continuum[:-1], color='k', lw=0.5)
        # ax[0].plot(xknot, yknot, '.', color='k')
        ax[1].plot(wl[ii][:-1], fl[ii][:-1]/continuum[:-1])
        ymax = np.max([ymax, np.quantile(fl[ii][:-1], 0.99)])
        ax[0].set_ylim(0, ymax)
        ax[1].set_ylim(0.65, 1.05)
    plt.show()

    fig, ax = plt.subplots(figsize=(18,10))
    ymax = 0
    for ii in np.arange(wl.shape[0]):
        continuum3, xknot, yknot = get_continuum_iterative(wl[ii], fl[ii], pstep=pstep, k=3, niters=0)
        continuum5, xknot, yknot = get_continuum_iterative(wl[ii], fl[ii], pstep=pstep, k=5, niters=0)
        # Other comparisons are pstep=60
        continuum3_half, xknot_half, yknot_half = get_continuum_iterative(wl[ii], fl[ii], pstep=200, k=3, niters=0)
        continuum5_half, xknot_half, yknot_half = get_continuum_iterative(wl[ii], fl[ii], pstep=200, k=5, niters=0)
        ax.plot(wl[ii][:-1], fl[ii][:-1])
        ax.plot(wl[ii][:-1], continuum3[:-1], color='k', ls='--', lw=1)
        ax.plot(wl[ii][:-1], continuum5[:-1], color='k', lw=1)
        ax.plot(wl[ii][:-1], continuum3_half[:-1], color='r', ls='--', lw=1)
        ax.plot(wl[ii][:-1], continuum5_half[:-1], color='r', lw=1)
        ax.plot(xknot, yknot, '.', color='k')
        ax.plot(xknot_half, yknot_half, '.', color='r')
        ymax = np.max([ymax, np.quantile(fl[ii][:-1], 0.99)])
        ax.set_ylim(0, ymax)
    plt.title('color=# of knots, dash=k3, solid=k5')
    plt.show()

    fig, ax = plt.subplots(figsize=(18,10))
    ymax = 0
    for ii in np.arange(wl.shape[0]):
        continuum3, xknot, yknot = get_continuum_iterative(wl[ii], fl[ii], pstep=pstep, k=3, niters=0)
        continuum5, xknot, yknot = get_continuum_iterative(wl[ii], fl[ii], pstep=pstep, k=5, niters=0)
        continuum3_half, xknot_half, yknot_half = get_continuum_iterative(wl[ii], fl[ii], pstep=200, k=3, niters=0)
        continuum5_half, xknot_half, yknot_half = get_continuum_iterative(wl[ii], fl[ii], pstep=200, k=5, niters=0)
        ax.plot(wl[ii][:-2], np.diff(continuum3[:-1])/np.diff(wl[ii][:-1]), color='k', ls='--', lw=1)
        ax.plot(wl[ii][:-2], np.diff(continuum5[:-1])/np.diff(wl[ii][:-1]), color='k', lw=1)
        ax.plot(wl[ii][:-2], np.diff(continuum3_half[:-1])/np.diff(wl[ii][:-1]), color='r', ls='--', lw=1)
        ax.plot(wl[ii][:-2], np.diff(continuum5_half[:-1])/np.diff(wl[ii][:-1]), color='r', lw=1)
        # ax.plot(xknot, yknot, '.', color='k')
        # ax.plot(xknot_half, yknot_half, '.', color='r')
        # ymax = np.max([ymax, np.quantile(fl[ii][:-1], 0.99)])
        # ax.set_ylim(0, ymax)
    plt.title('color=# of knots, dash=k3, solid=k5')
    plt.show()

    
