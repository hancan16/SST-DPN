import os
from typing import Dict

import numpy as np
import resampy
import scipy
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from numpy import multiply
from scipy.io import loadmat
from scipy.signal import filtfilt
from sklearn.preprocessing import StandardScaler


def z_scale(X, X_test):
    for ch_idx in range(X.shape[1]):
        sc = StandardScaler()
        X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
        X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
    return X, X_test


def load_bcic(
    dataset_id: str = "2a",
    subject_id: int = 1,
    preprocessing_dict: Dict = None,
    verbose: str = "WARNING",
):
    dataset_name = "BNCI2014_001" if dataset_id == "2a" else "BNCI2014_004"
    dataset = MOABBDataset(dataset_name, subject_ids=[subject_id])

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(lambda data: multiply(data, 1e6)),
    ]

    # filtering or not
    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(
            Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=verbose)
        )

    # resample or not
    if dataset.datasets[0].raw.info["sfreq"] != preprocessing_dict["sfreq"]:
        preprocessors.append(
            Preprocessor(
                "resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose
            ),
        )

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=False,
    )

    # split the data
    splitted_ds = windows_dataset.split("session")
    if dataset_id == "2a":
        # train_dataset已经是torch.utils.data.Dataset的实例
        train_dataset, test_dataset = (
            splitted_ds["0train"],
            splitted_ds["1test"],
        )

        # load the data
        X = np.stack([sample[0] for sample in train_dataset], axis=0)
        y = np.stack([sample[1] for sample in train_dataset], axis=0)
        X_test = np.stack([sample[0] for sample in test_dataset], axis=0)
        y_test = np.stack([sample[1] for sample in test_dataset], axis=0)

    elif dataset_id == "2b":
        train_datasets = [splitted_ds[f"{session}train"] for session in [0, 1, 2]]
        test_datasets = [splitted_ds[f"{session}test"] for session in [3, 4]]
        # load the data
        X_sess, y_sess, X_test_sess, y_test_sess = [], [], [], []
        for train_dataset in train_datasets:
            X_sess.append(np.stack([sample[0] for sample in train_dataset], axis=0))
            y_sess.append(np.stack([sample[1] for sample in train_dataset], axis=0))
        for test_dataset in test_datasets:
            X_test_sess.append(np.stack([sample[0] for sample in test_dataset], axis=0))
            y_test_sess.append(np.stack([sample[1] for sample in test_dataset], axis=0))

        X = np.concatenate(X_sess)
        y = np.concatenate(y_sess)
        X_test = np.concatenate(X_test_sess)
        y_test = np.concatenate(y_test_sess)
    if preprocessing_dict["z_scale"]:
        X, X_test = z_scale(X, X_test)
    return X, y, X_test, y_test


def bandpass_cheby2(data, low_cut_hz, high_cut_hz, fs, n=6, rs=60):
    b, a = scipy.signal.cheby2(
        N=n,
        rs=rs,
        Wn=[low_cut_hz, high_cut_hz],
        btype="bandpass",
        analog=False,
        output="ba",
        fs=fs,
    )
    data_bandpassed = filtfilt(
        b, a, data, axis=-1, padlen=3 * (max(len(b), len(a)) - 1)
    )
    return data_bandpassed


# BCIC 3
def load_bci3(dataPath, subject_id, preprocessing_dict):
    sub = {1: "aa", 2: "al", 3: "av", 4: "aw", 5: "ay"}
    path = os.path.join(
        dataPath,
        f"data_set_IVa_{sub[subject_id]}_mat",
        "100Hz",
        f"data_set_IVa_{sub[subject_id]}.mat",
    )
    label_path = os.path.join(dataPath, f"true_labels_{sub[subject_id]}.mat")
    mat = loadmat(path)
    mat_labels = loadmat(label_path)
    data = mat["cnt"].T
    marker = mat["mrk"][0][0][0]
    labels = mat_labels["true_y"]
    test_idx = mat_labels["test_idx"]

    sfreq = mat["nfo"]["fs"][0][0][0][0]
    ch_names = [_[0] for _ in mat["nfo"]["clab"][0][0][0]]

    channels = ["C3", "Cz", "C4"]
    channel_selection = preprocessing_dict.get("channel_selection", False)
    if channel_selection:
        channels_indices = [ch_names.index(ch) for ch in channels]
        data = data[channels_indices, :]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        data = bandpass_cheby2(data, l_freq, h_freq, sfreq)

    trial_length_second = preprocessing_dict["stop"] - preprocessing_dict["start"]

    start = int(sfreq * preprocessing_dict["start"])
    stop = int(sfreq * preprocessing_dict["stop"])
    trial_length = stop - start
    trials = np.zeros((labels.shape[-1], data.shape[0], trial_length))
    for i, m in enumerate(marker[0]):
        trials[i, ::] = data[:, m + start : m + stop]

    if preprocessing_dict["sfreq"] != sfreq:
        x = np.zeros(
            (
                trials.shape[0],
                trials.shape[1],
                int(preprocessing_dict["sfreq"] * trial_length_second),
            ),
            np.float32,
        )
        for i in range(trials.shape[0]):  # resampy.resample cant handle the 3D data.
            x[i, :, :] = resampy.resample(
                trials[i, :, :], sfreq, preprocessing_dict["sfreq"], axis=1
            )
        trials = x
    X, X_test = trials[: test_idx[0, 0] - 1], trials[test_idx[0] - 1]
    y, y_test = labels[0, : test_idx[0, 0] - 1] - 1, labels[0, test_idx[0] - 1] - 1

    if preprocessing_dict["z_scale"]:
        X, X_test = z_scale(X, X_test)
    else:
        X = normalize_data_per_sample(X)
        X_test = normalize_data_per_sample(X_test)
    return X, y, X_test, y_test


def normalize_data_per_sample(data):
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)

    normalized_data = (data - mean) / std

    return normalized_data


if __name__ == "__main__":
    # # * BCIC 2a
    preprocessing_2a = {
        "sfreq": 250,
        "low_cut": None,
        "high_cut": None,
        "start": 0,
        "stop": 0,
        "z_scale": False,
    }
    for i in range(1, 10):
        X, y, X_test, y_test = load_bcic("2a", i, preprocessing_2a)
        print(X.shape, y.shape, X_test.shape, y_test.shape)

    # * BCIC 2b
    # preprocessing_2b = {
    #     "sfreq": 250,
    #     "low_cut": None,
    #     "high_cut": None,
    #     "start": 0,
    #     "stop": 0,
    #     "z_scale": False,
    # }
    # for i in range(1, 10):
    #     X, y, X_test, y_test = load_bcic("2b", i, preprocessing_2b)
    #     print(X.shape, y.shape, X_test.shape, y_test.shape)

    # *BCI3
    # raw_path = "../data//BCIC3"
    # preprocessing_bci3 = {
    #     "sfreq": 100,
    #     "low_cut": None,
    #     "high_cut": None,
    #     "start": 0,
    #     "stop": 3.5,
    #     "z_scale": False,
    # }
    # for i in range(1, 6):
    #     X, y, X_test, y_test = load_bci3(raw_path, i, preprocessing_bci3)
    #     print(X.shape, y.shape, X_test.shape, y_test.shape)
