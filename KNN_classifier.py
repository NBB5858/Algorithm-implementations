import numpy as np


def k_smallest(x, k):
    # Given an input list x, returns the indices of the K smallest elements

    min_id = [None for i in range(k)]

    for i, x_el in enumerate(x):
        for j, min_id_el in enumerate(min_id):
            if (min_id_el is None) or x_el < x[min_id_el]:
                min_id.pop()
                min_id.insert(j, i)
                break

    return min_id


def compute_distances(a, b):
    # Computes Euclidean distances between pairs of row vectors comprising array A
    # and column vectors comprising array B

    a_square_mags = np.sum(a ** 2, axis=1, keepdims=True)
    b_square_mags = np.sum(b ** 2, axis=0, keepdims=True)

    a_dot_b = np.dot(a, b)

    square_dists = a_square_mags + b_square_mags - 2 * a_dot_b

    dists = np.sqrt(square_dists)

    return dists


def KNN_classifier(x, y, x_test, k):
    # KNN classifier
    # x is array comprised of row vectors of training data
    # y is 1 dimensional array of training data labels
    # x_test is array comprised of row vectors of test data

    if y.shape[0] != x.shape[0]:
        raise ValueError(f'different number of training labels ({y.shape[0]}) than training data ({x.shape[0]})')

    if k > x.shape[0]:
        k = x.shape[0]

    num_test_points = x_test.shape[0]

    classes = list(set(y))

    probs = np.zeros((num_test_points, len(classes)))
    classification = np.full(num_test_points, 'acdefghijkl')
    nearest_neighbors = np.zeros((num_test_points, k), dtype=int)
    neighbor_classes = np.full((num_test_points, k), 'abcdefghijkl')

    distances = compute_distances(x, np.transpose(x_test))

    # for each test point, determine neighbors and their classes, compute probabilities, and classify
    for i in range(num_test_points):
        nearest_neighbors[i, :] = k_smallest(distances[:, i], k)
        neighbor_classes[i, :] = y[nearest_neighbors[i, :]]

        for j, el in enumerate(classes):
            probs[i, j] = sum([1 for idx in range(k) if neighbor_classes[i, idx] == el]) / k

    classification = np.array(classes)[np.argmax(probs, axis=1).astype(int)]

    return classes, probs, classification
