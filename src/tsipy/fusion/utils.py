import numpy as np

__all__ = [
    "build_and_concat_label_mask",
    "build_and_concat_label_mask_output",
]


def build_and_concat_label_mask(x: np.ndarray, label: int) -> np.ndarray:
    """Builds a 1D label mask and concatenates it to x.

     Examples:
        >>> import numpy as np
        >>> a = np.array([0, 1, 2, 4])
        >>> build_and_concat_label_mask(a, label=1)
        array([[0, 1],
               [1, 1],
               [2, 1],
               [4, 1]])
        >>> b = np.array([[0, 1, 2, 4]])
        >>> build_and_concat_label_mask(a, label=2)
        array([[0, 2],
               [1, 2],
               [2, 2],
               [4, 2]])
        >>> c = np.array([[0, 1], [2, 4]])
        >>> build_and_concat_label_mask(c, label=3)
        array([[0, 1, 3],
               [2, 4, 3]])

    Returns:
        2D array with concatenated label mask.
    """
    if not isinstance(label, int):
        raise TypeError(f"Input label must be an int but is a {type(label)}")
    if label <= 0:
        raise ValueError(f"Input label must be a positive int: {label} > 0.")

    return _build_and_concat_label_mask(x, label)


def build_and_concat_label_mask_output(x: np.ndarray) -> np.ndarray:
    """Builds a 1D label mask and concatenates it to x. Output label is always -1.

     Examples:
        >>> import numpy as np
        >>> a = np.array([0, 1, 2, 4])
        >>> build_and_concat_label_mask_output(a)
        array([[ 0, -1],
               [ 1, -1],
               [ 2, -1],
               [ 4, -1]])

    Returns:
        2D array with concatenated label mask.
    """
    return _build_and_concat_label_mask(x, label=-1)


def _build_and_concat_label_mask(x: np.ndarray, label: int) -> np.ndarray:
    if x.ndim > 2 or len(x.shape) > 2:
        raise ValueError(f"Input x must be 1D or 2D array, but has shape {x.shape}")
    if len(x.shape) == 1:
        x = np.reshape(x, newshape=(-1, 1))

    labels = np.ones((x.shape[0], 1), dtype=np.int) * label  # x is 2D (n, k)
    x_labels = np.concatenate((x, labels), axis=1)  # x_labels is 2D (n, k + 1)

    assert (
        x_labels.ndim == 2 and len(x_labels.shape) == 2
    ), "x_labels with shape {} is not 2D.".format(x_labels.shape)
    assert (
        x_labels.shape[0] == x.shape[0]
    ), "Input and output must have same first dimension: {} != {}".format(
        x_labels.shape, x.shape
    )

    return x_labels
