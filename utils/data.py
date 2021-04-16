import numpy as np
from keras.datasets import mnist

def create_data(file_name, classes):
    '''
    Donwload the MNIST data and select only the two classes specified in 'classes'

    file_name: `string` name of the file to save the data to
    classes: `list` of `int`, len(classes) = 2 
    '''
    (train_X, train_y), (test_X, test_y) = mnist.load_data() # Download data

    # Convert to numpy array and normalise (= pixel range is 0-255 so divide by 255)
    train_X, train_y = np.array(train_X)/255, np.array(train_y).astype(np.int8)
    test_X, test_y = np.array(test_X)/255, np.array(test_y).astype(np.int8)

    # Select classes
    train_X = train_X[(train_y==classes[0]) | (train_y==classes[1])]
    train_y = train_y[(train_y==classes[0]) | (train_y==classes[1])]

    test_X = test_X[(test_y==classes[0]) | (test_y==classes[1])]
    test_y = test_y[(test_y==classes[0]) | (test_y==classes[1])]

    # Turn labels into {-1, 1}
    train_y[train_y == classes[1]] = 1
    train_y[train_y == classes[0]] = -1
    test_y[test_y == classes[1]] = 1
    test_y[test_y == classes[0]] = -1

    # Flatten images
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    np.savez_compressed(file_name, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
    
def img_flat_to_2d(flat_image):
    '''
    Used for visualising images, converts from a flattened array to a imx x imy array
    '''
    return flat_image.reshape(28, 28)