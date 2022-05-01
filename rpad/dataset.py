import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
import torch.utils.data as data_utils
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from .preprocessing import fill_missing_data, reconstruct_data, standardize_time_series

prefix = "./nasa-smd_processed/"


class KPIBatchedWindowDataset(Dataset):
    def __init__(self, series, label, mask, window_size=120, stride=1, use_pred=False, pred_steps=5, use_birnn=False,
                 birnn_method2=True):
        super(KPIBatchedWindowDataset, self).__init__()
        self.series = series
        self.label = label
        self.mask = mask

        self.window_size = window_size
        self.stride = stride
        self.use_pred = use_pred
        self.pred_steps = pred_steps
        self.use_birnn = use_birnn
        self.birnn_method2 = birnn_method2

        if len(self.series.shape) == 1:
            raise ValueError('The `series` must be an Multi array!')

        if label is not None and (label.shape[0] != series.shape[0]):
            raise ValueError('The shape of `label` must agrees with the shape of `series`!')

        if mask is not None and (mask.shape != series.shape):
            raise ValueError('The shape of `mask` must agrees with the shape of `series`!')

        # self.tails = np.arange(window_size, series.shape[0] + 1, stride)
        if use_birnn:
            if birnn_method2:
                self.tails = np.arange(window_size, series.shape[0] + 1 - window_size, stride)
            else:
                self.tails = np.arange(window_size + pred_steps, series.shape[0] + 1 - pred_steps, stride)
        else:
            self.tails = np.arange(window_size, series.shape[0] + 1 - pred_steps, stride)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: x:multi_series of a window_size, y_pred: after this window,the next predict_step window , also multi_series
        """
        if self.birnn_method2:
            x = self.series[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
            x_next = self.series[self.tails[idx]:self.tails[idx] + self.window_size].astype(np.float32)
        else:
            x = self.series[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

        if (self.label is None) and (not self.use_pred):
            # Only data
            return torch.from_numpy(x)

        elif (self.label is not None) and (not self.use_pred):
            # Data and label
            y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)

        elif (self.label is None) and (self.use_pred):
            # Data and mask
            if self.use_birnn:
                if self.birnn_method2:
                    y_pred_forward = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                    y_pred_backward = self.series[self.tails[idx] - self.pred_steps:self.tails[idx]].astype(np.float32)
                    return torch.from_numpy(x), torch.from_numpy(x_next), torch.from_numpy(
                        y_pred_forward), torch.from_numpy(y_pred_backward)
                else:
                    y_pred_forward = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                    y_pred_backward = self.series[self.tails[idx] - self.window_size - self.pred_steps:self.tails[
                                                                                                           idx] - self.window_size].astype(
                        np.float32)
                    return torch.from_numpy(x), torch.from_numpy(y_pred_forward), torch.from_numpy(y_pred_backward)
            else:
                y_pred = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                return torch.from_numpy(x), torch.from_numpy(y_pred)

        elif (self.label is not None) and (self.use_pred):
            if self.use_birnn:
                if self.birnn_method2:
                    y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
                    y_pred_forward = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                    y_pred_backward = self.series[self.tails[idx] - self.pred_steps:self.tails[idx]].astype(np.float32)
                    return torch.from_numpy(x), torch.from_numpy(x_next), torch.from_numpy(y), torch.from_numpy(
                        y_pred_forward), torch.from_numpy(y_pred_backward)
                else:
                    y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
                    y_pred_forward = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                    y_pred_backward = self.series[self.tails[idx] - self.window_size - self.pred_steps:self.tails[
                                                                                                           idx] - self.window_size].astype(
                        np.float32)
                    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_pred_forward), torch.from_numpy(
                        y_pred_backward)
            else:
                y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
                y_pred = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_pred)

    def __len__(self):
        return self.tails.shape[0]


class KPIBatchedWindowDataset2(Dataset):
    """
    """

    def __init__(self, series, label, mask, window_size=120, stride=1, use_pred=False, pred_steps=5, use_birnn=False,
                 birnn_method2=True):
        super(KPIBatchedWindowDataset2, self).__init__()
        self.series = series
        self.label = label
        self.mask = mask

        self.window_size = window_size
        self.stride = stride
        self.use_pred = use_pred
        self.pred_steps = pred_steps
        self.use_birnn = use_birnn
        self.birnn_method2 = birnn_method2

        if len(self.series.shape) == 1:
            raise ValueError('The `series` must be an Multi array!')

        if label is not None and (label.shape[0] != series.shape[0]):
            raise ValueError('The shape of `label` must agrees with the shape of `series`!')

        if mask is not None and (mask.shape != series.shape):
            raise ValueError('The shape of `mask` must agrees with the shape of `series`!')

        # self.tails = np.arange(window_size, series.shape[0] + 1, stride)
        if use_birnn:
            self.tails = np.arange(window_size + pred_steps, series.shape[0] + 1 - pred_steps, stride)
        else:
            self.tails = np.arange(window_size, series.shape[0] + 1 - pred_steps, stride)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: x:multi_series of a window_size, y_pred: after this window,the next predict_step window , also multi_series
        """
        x = self.series[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)

        if (self.label is None) and (not self.use_pred):
            # Only data
            # return torch.from_numpy(x)
            return torch.from_numpy(x), torch.from_numpy(x)

        elif (self.label is not None) and (not self.use_pred):
            # Data and label
            y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)

        elif (self.label is None) and (self.use_pred):
            # Data and mask
            if self.use_birnn:
                y_pred_forward = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                y_pred_backward = self.series[self.tails[idx] - self.window_size - self.pred_steps:self.tails[
                                                                                                       idx] - self.window_size].astype(
                    np.float32)
                return torch.from_numpy(x), torch.from_numpy(y_pred_forward), torch.from_numpy(y_pred_backward)
            else:
                y_pred = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                return torch.from_numpy(x), torch.from_numpy(y_pred)

        elif (self.label is not None) and (self.use_pred):
            if self.use_birnn:
                y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
                y_pred_forward = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                y_pred_backward = self.series[self.tails[idx] - self.window_size - self.pred_steps:self.tails[
                                                                                                       idx] - self.window_size].astype(
                    np.float32)
                return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_pred_forward), torch.from_numpy(
                    y_pred_backward)
            else:
                y = self.label[self.tails[idx] - self.window_size: self.tails[idx]].astype(np.float32)
                y_pred = self.series[self.tails[idx]:self.tails[idx] + self.pred_steps].astype(np.float32)
                return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_pred)

    def __len__(self):
        return self.tails.shape[0]


def prepare_dataset(data_path, data_category, train_val_test_split, label_portion, standardization_method='negpos1',
                    filling_method='zero'):
    # Assertions
    assert (sum(train_val_test_split) == 10)
    assert (standardization_method in ['standard', 'minmax', 'negpos1', 'none'])
    assert (filling_method in ['prev', 'zero', 'none'])

    with open(data_path, 'r', encoding='utf8') as f:
        df = pd.read_csv(f)
    if data_category == 'kpi':
        time_stamp = df['timestamp'].values
        value = df['value'].values
        label = df['label'].values
    elif data_category == 'nab':
        time_stamp = np.array(
            list(map(lambda s: time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S')), df['timestamp'].values)))
        value = df['value'].values
        label = df['label'].values
    elif data_category == 'yahoo':
        if 'timestamp' in df.columns:
            time_stamp = df['timestamp'].values
        else:
            time_stamp = df['timestamps'].values
        value = df['value'].values
        if 'changepoint' in df.columns:
            label = np.logical_or(df['changepoint'].values, df['anomaly'].values)
        else:
            label = df['is_anomaly'].values
    else:
        raise ValueError('Invalid data category!')

    # Reconstruct data
    time_stamp, value, label, mask = reconstruct_data(time_stamp, value, label)
    # Filling missing data
    value = fill_missing_data(value, mask, method=filling_method)
    # Standardization
    value = standardize_time_series(value, standardization_method, mask=np.logical_or(label, mask))

    # datetimes = [datetime.fromtimestamp(time_stamp[i]) for i in range(len(time_stamp))]

    # Pre-processing
    quantile1 = train_val_test_split[0] / 10
    quantile2 = (10 - train_val_test_split[-1]) / 10

    train_x, train_y, train_m = value[:int(value.shape[0] * quantile1)], \
                                label[:int(label.shape[0] * quantile1)], \
                                mask[:int(mask.shape[0] * quantile1)]
    val_x, val_y, val_m = value[int(value.shape[0] * quantile1):int(value.shape[0] * quantile2)], \
                          label[int(label.shape[0] * quantile1):int(label.shape[0] * quantile2)], \
                          mask[int(mask.shape[0] * quantile1):int(mask.shape[0] * quantile2)]
    test_x, test_y, test_m = value[int(value.shape[0] * quantile2):], \
                             label[int(label.shape[0] * quantile2):], \
                             mask[int(mask.shape[0] * quantile2):]

    if quantile1 == quantile2:
        val_x = None
        val_y = None

    if label_portion == 0.0:
        train_y = np.zeros_like(train_y)
    else:
        anomaly_indices = np.arange(train_y.shape[0])[train_y == 1]
        selected_indices = np.random.choice(anomaly_indices,
                                            size=int(np.floor(anomaly_indices.shape[0] * (1 - label_portion))),
                                            replace=False)
        train_y[selected_indices] = 0

    return train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m


def multi_get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def smd_interpretation_testdata(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True,
                                train_start=0,
                                test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)

    x_dim = multi_get_data_dim(dataset)
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        test_data = multi_preprocess(test_data)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    test_interpretation1 = test_data[11310: 11516]
    test_interpretation2 = test_data[22923: 23019]

    test_label1 = test_label[11310: 11516]
    test_label2 = test_label[22923: 23019]
    return (test_data, test_label, test_interpretation1, test_interpretation2, test_label1, test_label2)


def multi_get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
                   test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = multi_get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = multi_preprocess(train_data)
        test_data = multi_preprocess(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def swat_multi_get_data(dataset):
    # Normal
    normal = pd.read_csv("swat/SWaT_Dataset_Normal_v1.csv")  # , nrows=1000)
    normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
    # print(normal.shape)

    # Transform all columns into float64
    for i in list(normal):
        normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
    normal = normal.astype(float)

    # Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x = normal.values
    x_scaled = min_max_scaler.fit_transform(x)
    normal = pd.DataFrame(x_scaled)
    # print(normal.head())

    # Attack
    # Read data
    attack = pd.read_csv("swat/SWaT_Dataset_Attack_v0.csv", sep=";")  # , nrows=1000)
    labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
    attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
    # print("label_shape:::::")
    # print(len(labels))
    # print(attack.shape)

    # Transform all columns into float64
    for i in list(attack):
        attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
    attack = attack.astype(float)

    # Normalization
    x = attack.values
    x_scaled = min_max_scaler.transform(x)
    attack = pd.DataFrame(x_scaled)

    return (normal.values, None), (attack.values, np.array(labels))


def multi_preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df
