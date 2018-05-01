from keras.datasets import mnist

import scipy.io as sio
import urllib.request
import shutil
import os
import numpy as np

def get_mnist():
   
   (train_x, train_y), (test_x, test_y) = mnist.load_data()
   train_x = np.stack([train_x]*3, axis=3)
   test_x = np.stack([test_x]*3, axis=3)
   
   return (train_x, train_y), (test_x, test_y) 

def get_svhn():
    
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    
    
    if not os.path.exists(os.path.join('datasets', 'svhn_train.mat')):
        print ('Downloading SVHN training set!')
        with urllib.request.urlopen('http://ufldl.stanford.edu/housenumbers/train_32x32.mat') as response, open(os.path.join('datasets', 'svhn_train.mat'), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    
    if not os.path.exists(os.path.join('datasets', 'svhn_test.mat')):
        print ('Downloading SVHN test set!')
        with urllib.request.urlopen('http://ufldl.stanford.edu/housenumbers/test_32x32.mat') as response, open(os.path.join('datasets', 'svhn_test.mat'), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    
    train = sio.loadmat(os.path.join('datasets', 'svhn_train.mat'))
    test = sio.loadmat(os.path.join('datasets', 'svhn_test.mat'))
    
    return (np.transpose(train['X'], (3,0,1,2)), train['y'].flatten()), (np.transpose(test['X'], (3,0,1,2)), test['y'].flatten()) 

if __name__=='__main__':

    (train_x, train_y), (test_x, test_y) = get_svhn()
    print (train_x.shape, train_y.shape, test_x.shape, test_y)

    (train_x, train_y), (test_x, test_y) = get_mnist()
    print (train_x.shape, train_y.shape, test_x.shape, test_y)
