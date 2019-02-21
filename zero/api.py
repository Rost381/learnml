from . import supervised
from .supervised.decision_tree import ClassificationTree, RegressionTree
from .supervised.linear_discriminant_analysis import LinearDiscriminantAnalysis
from .supervised.regression import LinearRegression, RidgeRegression, LassoRegression
from .supervised.support_vector_machine_cvxopt import svmCVXOPT

from . import unsupervised
from .unsupervised.principal_component_analysis import PCA

from . import deep
