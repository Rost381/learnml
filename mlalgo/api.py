from . import supervised
from .supervised.decision_tree import ClassificationTree, RegressionTree
from .supervised.linear_discriminant_analysis import LinearDiscriminantAnalysis
from .supervised.regression import LinearRegression, RidgeRegression, LassoRegression
from .supervised.support_vector_machine import SVM

from . import unsupervised
from .unsupervised.principal_component_analysis import PCA

from . import deep

from . import utils
from .utils.tools import covariance_matrix, calculate_variance, calculate_entropy, standardize, normalize
from .utils.tools import calculate_accuracy_score, calculate_mean_squared_error
from .utils.tools import divide_on_feature
