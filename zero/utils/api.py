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
from .tools import divide_on_feature

from .plot import make_meshgrid, plot_contours

from .preprocessing import PolynomialFeatures

from .activation_functions import Sigmoid

from .loss_function import (
    l1_regularization,
    l2_regularization,
    l1_loss,
    l2_loss,
    cross_entropy_loss
)

from .np import to_categorical
