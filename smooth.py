import numpy as np

# Function to smooth the input array using a specific window
def smooth(x, window_len=25, window='blackman'):
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    if x.size < window_len:
        raise ValueError("Input size must be larger than the window size.")

    if window_len < 3:
        return x

    # Supported windows
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(f"Unsupported window type: {window}")

    # Create extended array for convolution
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    # Select the window function
    if window == 'flat':
        w = np.ones(window_len, dtype='d')
    else:
        w = eval(f'np.{window}(window_len)')

    # Perform convolution
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

# Function to smooth the luminance percentile curves of video frames
def smoothen_luminance(predictions, percs):
    percs = np.array(percs, dtype='float32').transpose()  # Convert to NumPy array and transpose
    smooth_percs = []

    # Smooth each percentile curve
    for perc in percs:
        smooth_percs.append(smooth(perc))

    # Transpose smoothed percentiles back
    smooth_percs = np.array(smooth_percs).transpose()

    # Interpolate predictions to the smoothed percentiles
    smoothed_predictions = []
    for i, pred in enumerate(predictions):
        smooth_pred = np.interp(pred, percs[i], smooth_percs[i]).astype('float32')
        smoothed_predictions.append(smooth_pred.clip(0, 1))  # Clip values between 0 and 1

    return smoothed_predictions
