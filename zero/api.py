from . import supervised
from .supervised.decision_tree import ClassificationTree, RegressionTree
from .supervised.linear_discriminant_analysis import LinearDiscriminantAnalysis
from .supervised.regression import LinearRegression, RidgeRegression, LassoRegression, PolynomialRidgeRegression
from .supervised.logistic_regression import LogisticRegression
from .supervised.support_vector_machine import svm
from .supervised.adaboost import AdaBoostClassifier

from . import unsupervised
from .unsupervised.principal_component_analysis import PCA

from . import deep
