# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import subprocess
import pickle
import torch
import os

cifar_path = "cifar-100-python.tar.gz"
cifar10_path = "cifar-10-python.tar.gz"
mnist_path = "mnist.npz"

# URL from: https://www.cs.toronto.edu/~kriz/cifar.html
if not os.path.exists(cifar_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", shell=True)

subprocess.call("tar xzfv cifar-100-python.tar.gz", shell=True)

# URL from: https://www.cs.toronto.edu/~kriz/cifar.html
if not os.path.exists(cifar10_path):
    subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", shell=True)

subprocess.call("tar xzfv cifar-10-python.tar.gz", shell=True)

# URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
if not os.path.exists(mnist_path):
    subprocess.call("wget https://s3.amazonaws.com/img-datasets/mnist.npz", shell=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


cifar100_train = unpickle('cifar-100-python/train')
cifar100_test = unpickle('cifar-100-python/test')

x_tr = torch.from_numpy(cifar100_train[b'data'])
y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
x_te = torch.from_numpy(cifar100_test[b'data'])
y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar100.pt')

x_tr=None
for batch in range(5):#only two batches
    cifar10_train = unpickle('cifar-10-batches-py/data_batch_'+str(batch+1))
    if x_tr is None:
        x_tr = torch.from_numpy(cifar10_train[b'data'])
        y_tr = torch.LongTensor(cifar10_train[b'labels'])
    else:
        x_tr = torch.cat((x_tr,torch.from_numpy(cifar10_train[b'data'])),0)
        y_tr = torch.cat((y_tr,torch.LongTensor(cifar10_train[b'labels'])),0)

cifar10_test = unpickle('cifar-10-batches-py/test_batch')
print("cifar 10 train size is ",y_tr.size(0))

x_te = torch.from_numpy(cifar10_test[b'data'])

y_te = torch.LongTensor(cifar10_test[b'labels'])
x_te=x_te[0:1000]
y_te=y_te[0:1000]
torch.save((x_tr, y_tr, x_te, y_te), 'cifar10.pt')


f = np.load('mnist.npz')
x_tr = torch.from_numpy(f['x_train'])
y_tr = torch.from_numpy(f['y_train']).long()
x_te = torch.from_numpy(f['x_test'])
y_te = torch.from_numpy(f['y_test']).long()
f.close()

torch.save((x_tr, y_tr), 'mnist_train.pt')
torch.save((x_te, y_te), 'mnist_test.pt')
