from . import deep, supervised, unsupervised
from .supervised.adaboost import AdaBoostClassifier
from .supervised.decision_tree import (ClassificationTree, RegressionTree,
                                       XGBoostRegressionTree)
from .supervised.gradient_boosting import (GradientBoostingClassifier,
                                           GradientBoostingRegressor)
from .supervised.k_nearest_neighbors import KNeighborsClassifier
from .supervised.linear_discriminant_analysis import LinearDiscriminantAnalysis
from .supervised.logistic_regression import LogisticRegression
from .supervised.naive_bayes import GaussianNB
from .supervised.random_forest import RandomForestClassifier
from .supervised.regression import (LassoRegression, LinearRegression,
                                    PolynomialRidgeRegression, RidgeRegression)
from .supervised.support_vector_machine import svm
from .supervised.xgboost import XGBoost
from .unsupervised.principal_component_analysis import PCA
