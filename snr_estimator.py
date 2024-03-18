# from alexmods.specutils.spectrum import read_mike_spectrum
from astropy.io import fits
# import matplotlib.pyplot as plt
import numpy as np
import argparse

def estimate_snr(file_name, arm='blue', read='fast', plot=False):
    """
    if file_name is a list, then stack.

    NOTE: this method was developed to estimate high SNR (>100)
    on the blue arm of MIKE. It does not really work at the
    reddest orders (between orders 34 - 50ish on the red side).
    """
    order_center_blue = [27, 63, 98, 133, 168, 201,
                         235, 268, 300, 332, 363, 394,
                         424, 454, 484, 513, 541, 569,
                         597, 624, 651, 678, 704, 730,
                         755, 780, 804, 829, 853, 876,
                         899, 922, 945, 967, 989, 1010]
    
    order_center_red = [44, 84, 124, 163, 201, 238,
                        274, 310, 344, 378, 411, 444,
                        476, 507, 537, 567, 596, 624,
                        652, 680, 706, 733, 758, 784,
                        808, 832, 856, 880, 903, 926,
                        948, 970, 992, 1013]

    try:
        data = fits.open(file_name)[0].data
    except:
        data = fits.open(file_name[0])[0].data
        for ii in np.arange(len(file_name))[1:]:
            data += fits.open(file_name[ii])[0].data

    if arm == 'blue':
        n_order = 36
        order_x = order_center_blue
        n_start = 72
        
        if read == 'fast':
            gain = 0.9
        if read == 'slow':
            gain = 0.45
    if arm == 'red':
        n_order = 34
        order_x = order_center_red
        n_start = 37
        
        if read == 'fast':
            gain = 0.9
        if read == 'slow':
            gain = 0.9

    snr_estimate = np.zeros(n_order)
    order_y = 1088 # middle of the detector

    order_x_refined = np.zeros(n_order)
    
    for ii, idx in enumerate(order_x):
        # Get a slightly better centering if needed (this seems mostly unnecessary).
        # Also, note that (confusingly) the shape of data is [y,x] not [x,y]
        didx = np.arange(-5, 5)[np.argmax(data[order_y, idx-5:idx+5])]
        order_slice = data[order_y, didx+idx-10:didx+idx+10]
        snr_estimate[ii] = np.sqrt(gain * (np.sum(order_slice - order_slice.min())))
        order_x_refined[ii] = idx+didx

    if plot:
        plt.figure()
        plt.plot(np.arange(len(data[order_y, :])), data[order_y, :], color='k')
        for x in order_x_refined:
            plt.plot(np.arange(x-10, x+10), data[order_y, int(x)-10:int(x)+10], color='r')
            plt.plot(x, data[order_y, int(x)], '.', color='b')
        plt.show()

    for ii in np.arange(n_order):
        print('Order {0} : SNR = {1:.0f}'.format(n_start + ii, snr_estimate[::-1][ii]))
                
    return snr_estimate


def compare_estimate_vs_actual_snr(estimate_file_name, actual_file_name, arm='blue', read='fast'):
    snr_estimate = estimate_snr(estimate_file_name, arm=arm, read=read)

    data = read_mike_spectrum(actual_file_name, fluxband=4)

    if arm == 'blue':
        n_order = 36
        n_start = 72
    if arm == 'red':
        n_order = 34
        n_start = 37
        
    snr_actual = np.zeros(n_order)
    
    for kk, key in enumerate(data.keys()):
        diff = np.concatenate([np.array([0]), np.diff(data[key].flux)])
        median, std = np.median(diff), np.std(diff)
        keep_idx = np.where((diff > median - 3*std) &
                            (diff < median + 3*std))[0]
        snr_actual[kk] = np.max(data[key].flux[keep_idx])
        
    plt.figure()
    # Also confusing, it goes backwards in the estimate...
    plt.plot(n_start + np.arange(n_order), snr_estimate[::-1], label='Estimate')
    plt.plot(n_start + np.arange(n_order), snr_actual, label='Actual')
    plt.legend()
    plt.xlabel('Order')
    plt.ylabel('SNR')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate the SNR of a raw spectrum"
    )
    parser.add_argument("file_name")
    parser.add_argument("--arm", choices=['red', 'blue'], default='blue')
    parser.add_argument("--read", choices=['fast', 'slow'], default='fast')
    parser.add_argument("--plot", choices=['True', 'False'], default=False)
    args = parser.parse_args()

    file_name = args.file_name
    arm = args.arm
    read = args.read
    plot = args.plot
    
    estimate_snr(file_name, arm=arm, read=read, plot=plot)
    
