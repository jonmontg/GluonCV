import numpy as np


def print_ary_props(ary: np.ndarray) -> None:
    print('shape: ', ary.shape)
    print('data type: ', ary.dtype)
    print('minimum value: ', ary.min().asscalar())
    print('maximum value: ', ary.max().asscalar())