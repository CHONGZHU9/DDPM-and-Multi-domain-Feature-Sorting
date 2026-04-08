import os
import numpy as np
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import librosa
from multiprocessing import Pool, cpu_count
import scipy.signal as signal

def hjorth_features(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1)

    return activity, mobility, complexity

def nonlinear_energy(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    nonlinear_energy = np.mean(signal**2 - np.convolve(signal, second_deriv, mode='same'))
    return nonlinear_energy

def spectral_correlation_coefficient(Pxx):
    autocorrelation = np.correlate(Pxx, Pxx, mode='full')
    correlation_coefficient = autocorrelation[len(Pxx) - 1] / np.sum(Pxx**2)
    return correlation_coefficient

def intensity_weighted_mean_frequency(f, Pxx):
    iwmf = np.sum(f * Pxx) / np.sum(Pxx)
    return iwmf

def intensity_weighted_bandwidth(f, Pxx):
    iwmf = intensity_weighted_mean_frequency(f, Pxx)
    iwbw = np.sqrt(np.sum(((f - iwmf) ** 2) * Pxx) / np.sum(Pxx))
    return iwbw

def extract_features(args):
    i, j, row = args
    features = np.empty(24)

    activity, mobility, complexity = hjorth_features(row)
    features[0] = activity
    features[1] = mobility
    features[2] = complexity

    sr = 256
    n_mfcc = 10
    frame_length = 1024
    hop_length = 512

    frames = librosa.util.frame(row, frame_length=frame_length, hop_length=hop_length)
    spectra = np.array([np.abs(librosa.stft(frame, n_fft=frame_length)) for frame in frames.T])
    mfccs = np.array([librosa.feature.mfcc(S=spectrum, sr=sr, n_mfcc=n_mfcc) for spectrum in spectra])

    mfccs_std = np.std(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)

    features[3:8] = mfccs_std
    features[8:13] = mfccs_var

    zcr = librosa.feature.zero_crossing_rate(row, frame_length=frame_length, hop_length=hop_length)
    features[13] = np.mean(zcr)

    nl_energy = nonlinear_energy(row)
    features[14] = nl_energy

    f, Pxx = signal.welch(row, fs=sr, nperseg=256, noverlap=128)

    features[15] = np.std(Pxx)
    features[16] = np.var(Pxx)

    peak_freq = f[np.argmax(Pxx)]
    features[17] = peak_freq

    bandwidth = np.sum(f ** 2 * Pxx) / np.sum(Pxx) - (np.sum(f * Pxx) / np.sum(Pxx)) ** 2
    features[18] = bandwidth

    features[19] = skew(Pxx)
    features[20] = kurtosis(Pxx)
    features[21] = spectral_correlation_coefficient(Pxx)
    features[22] = intensity_weighted_mean_frequency(f, Pxx)
    features[23] = intensity_weighted_bandwidth(f, Pxx)

    return i, j, features

def process_file(file_path):
    data = np.load(file_path)
    data_features = np.empty((data.shape[0], data.shape[1], 24))

    args_list = [(i, j, data[i, j, :]) for i in range(data.shape[0]) for j in range(data.shape[1])]

    with Pool(cpu_count()) as pool:
        results = pool.map(extract_features, args_list)

    for i, j, features in results:
        data_features[i, j, :] = features

    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_features.npy'
    output_file_path = os.path.join(output_path, output_file_name)
    np.save(output_file_path, data_features)

if __name__ == '__main__':
    input_path = "E:/EEG/EEGtest/reshape"
    output_path = "E:/EEG/EEGtest/features"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in tqdm(os.listdir(input_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_path, file_name)
            process_file(file_path)
