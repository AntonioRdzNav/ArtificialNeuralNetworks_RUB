import numpy as np


def sign(x):
    if x <= 0:
        return -1
    else:
        return 1


def sign_vec(x):
    return np.where(x <= 0, -1, 1)


def retrieve_async(x, w, num_iter, b=0):
    for ind in range(num_iter):
        j = np.random.randint(0, x.size)
        x[j] = sign(w[j, :].dot(x) + b)
    return x


def retrieve_sync(x, w, num_iter, b=0):
    for ind in range(num_iter):
        x = sign_vec(x.dot(w) + b)
    return x

# Eq. 2 from lecture notes
def calculate_weights(images):
    w = np.zeros((images.shape[0], images.shape[0]))
    for image in images.T:
        w += image * image[np.newaxis].T
    np.fill_diagonal(w, 0)
    return w / images.shape[1]
