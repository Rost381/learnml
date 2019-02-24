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