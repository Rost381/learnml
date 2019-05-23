from .stats import (
    covariance_matrix,
    calculate_variance,
    calculate_entropy,
    standardize,
    normalize
)
from .metrics import (
    calculate_accuracy_score,
    calculate_mean_squared_error,
)

from .pairwise import euclidean_distances

from .plot import make_meshgrid, plot_contours

from .preprocessing import PolynomialFeatures

from .activations import (
    Sigmoid, ReLU, SoftPlus, LeakyReLU, TanH, ELU, SELU, Softmax
)

from .losses import (
    l1_regularization,
    l2_regularization,
    l1_loss,
    l2_loss,
    cross_entropy_loss,
    logistic_loss
)

from .np import to_categorical

from .random import random_subsets, random_arr

from .deep import (
    determine_padding,
    get_im2col_indices,
    image_to_column,
    column_to_image
)