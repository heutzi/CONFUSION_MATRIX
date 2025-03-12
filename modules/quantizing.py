import numpy as np
import numpy.typing as npt


def to_label(
        class_distrib: npt.NDArray[np.float64],
        num_classes: int
        ) -> npt.NDArray[np.int_]:
    """Converts an array representing a ternary class distribution
    into an array of single values in range(0, 2*num_classes).

    First, the majority class between +1 and -1 is chosen.
    The probability space [0,1] is divided in num_classes ranges.
    Each probability is mapped to its corresponding range.
    -1 values are attributed the first classes,
        in reverse order
    +1 values are attributed the last num_classes classes.

    Args:
        class_distrib (npt.NDArray[np.float64]):
            Array representing a class distribution
        num_classes (int):
            Number of classes between which distributional values are divided

    Returns:
        npt.NDArray[np.int_]:
            Array of single values in range(0, 2*num_classes)
            representing the chosen class
    """
    def quantize(array: np.ndarray, rat: int):
        return np.floor(array*0.99*rat)+1

    bin_distrib = np.delete(class_distrib, 1, 1)
    is_pos_neg = np.argmax(bin_distrib, axis=1)
    quantized = quantize(
        bin_distrib[np.arange(len(bin_distrib)), is_pos_neg],
        num_classes
        )
    norm_quant = quantized * (2*is_pos_neg-1) + num_classes - is_pos_neg
    return norm_quant.astype(int)


def to_label_tern_prob(
        class_distrib: npt.NDArray[np.float64],
        threshold: float
        ) -> npt.NDArray[np.int_]:
    """Converts an array representing a ternary class distribution
    into an array of single values in {-1, 0, +1}.

    First, the majority class between +1 and -1 is chosen.
    When +1 is the majority class
        each value above the threshold is converted to +1.
    When -1 is the majority class
        each value above the threshold is converted to -1.
    The rest is converted to 0.

    Args:
        class_distrib (npt.NDArray[np.float64]):
            Array representing a class distribution
        threshold (float):
            Number between 0 and 1 representing the threshold
            at which -1 and +1 categories are chosen
    Returns:
        npt.NDArray[np.int_]:
            Array of ints in {-1, 0, +1} representing the chosen class
    """
    def quantize(
            array: npt.NDArray[np.float64],
            threshold: float
            ) -> npt.NDArray[np.int_]:
        sup, inf = array >= threshold, array <= -threshold
        array[sup] = 1
        array[inf] = -1
        array[np.logical_and(~sup, ~inf)] = 0
        return array

    bin_distrib = np.delete(class_distrib, 1, 1)
    is_pos_neg = np.argmax(bin_distrib, axis=1)
    bin_distrib[:, 0] *= -1
    quantized = quantize(
        bin_distrib[np.arange(len(bin_distrib)), is_pos_neg],
        threshold
        )
    return quantized.astype(int)


def to_label_tern_class(
        class_distrib: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.int_]:
    """Converts an array representing a ternary class distribution
    into an array of single values in {-1, 0, +1}

    Each distribution is mapped to the majority class.

    Args:
        class_distrib (npt.NDArray[np.float64]):
            Array representing a class distribution
    Returns:
        npt.NDArray[np.int_]:
            Array of ints in {-1, 0, +1} representing the chosen class
    """
    label_pred = np.argmax(class_distrib, axis=1)
    label_pred = label_pred-1
    return label_pred.astype(int)
