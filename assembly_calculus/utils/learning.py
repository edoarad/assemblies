from numpy.core._multiarray_umath import ndarray


def value_or_default(value, default):
    if value is None:
        return default
    return value


def get_matrix_min(matrix):
    """
    :param matrix: a 1D/2D list or a numpy object
    :type matrix: ndarray or (list of (list of float))
    :return: the minimal value of the matrix
    :rtype: float
    """
    if isinstance(matrix, ndarray):
        return matrix.min()
    elif isinstance(matrix, list):
        if all(isinstance(x, (int, float)) for x in matrix):
            return min(matrix)
        return min(min(row) for row in matrix)
    raise Exception('Unsupported type of matrix')


def get_matrix_max(matrix):
    """
    :param matrix: a 1D/2D list or a numpy object
    :type matrix: ndarray or (list of list)
    :return: the maximal value of the matrix
    :rtype: float
    """
    if isinstance(matrix, ndarray):
        return matrix.max()
    elif isinstance(matrix, list):
        if all(isinstance(x, (int, float)) for x in matrix):
            return max(matrix)
        return max(max(row) for row in matrix)
    raise Exception('Unsupported type of matrix')