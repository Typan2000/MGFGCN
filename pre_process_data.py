import numpy as np
import torch
import os
from scipy.stats import wasserstein_distance
import time

def pairwise_wasserstein(vectors):
    n = len(vectors)
    dist_matrix = np.zeros((n, n))
    t1 = time.time()
    for i in range(n):
        if (i % 10 == 1):
            t1 = time.time()
        for j in range(i + 1, n):
            dist_matrix[i, j] = wasserstein_distance(vectors[i], vectors[j])
            dist_matrix[j, i] = dist_matrix[i, j]
        if(i % 10 == 0):
            t2 = time.time()
            print('Line',i,'finished in',t2-t1,'seconds.')
    return dist_matrix

def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    dist_matrix = pairwise_wasserstein(data['data'][:train_line,:,0].T)
    dist_matrix = torch.from_numpy(dist_matrix)
    torch.save(dist_matrix,"data/graph/adj_sim_" + dataset + ".pt")
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_data(graph_signal_matrix_filename, transformer=None):
    data = np.load(graph_signal_matrix_filename)
    print("data type",type(data))
    print("data len",len(data))
    keys = data.keys()
    print("data keys",keys)

    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


seq_length_x =12
seq_length_y =12
dataset = 'PEMS08'

if __name__ == '__main__':
    x_list = []
    y_list = []

    for idx, (x, y) in enumerate(generate_data("./data/" + dataset + "/" + dataset + ".npz")):
        print(x.shape, y.shape)
        x_list.append(x)
        y_list.append(y)

    x_train = x_list[0].transpose(0, 3, 2, 1)
    y_train = y_list[0].transpose(0, 3, 2, 1)
    x_val = x_list[1].transpose(0, 3, 2, 1)
    y_val = y_list[1].transpose(0, 3, 2, 1)
    x_test = x_list[2].transpose(0, 3, 2, 1)
    y_test = y_list[2].transpose(0, 3, 2, 1)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join("data/" + dataset , f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )






