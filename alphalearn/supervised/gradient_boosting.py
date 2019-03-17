import numpy as np

from alphalearn.api import RegressionTree
from alphalearn.utils.api import cross_entropy_loss, l2_loss, softmax, to_categorical


class GradientBoosting():
    """Gradient Boosting base.

    Parameters:
    -----------
    n_estimators : int (default=100)
        The number of boosting stages to perform. 
        Gradient boosting is fairly robust to over-fitting so a large number
        usually results in better performance.
    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by learning_rate.
        There is a trade-off between learning_rate and n_estimators.
    min_samples_split : int, float
        The minimum number of samples required to split an internal node:
        If int, then consider min_samples_split as the minimum number.
        If float, then min_samples_split is a fraction and
        ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
    min_impurity_split : float
        Threshold for early stopping in tree growth.
        A node will split if its impurity is above the threshold, otherwise it is a leaf.
    max_depth : integer,
        maximum depth of the individual regression estimators.
    isClassifier : bool
        is Gradient Boosting Classifier?
    trees : list
        Regression Trees
    """

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=3,
                 _isClassifier=False,
                 _isRegressor=False
                 ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth
        self._isClassifier = _isClassifier
        self._isRegressor = _isRegressor
        self.trees = []

        """Loss function
        Classifier: cross_entropy_loss
        Regression: l2_loss
        """
        if self._isClassifier:
            self.loss_function = cross_entropy_loss()
        if self._isRegressor:
            self.loss_function = l2_loss()

        """Create many tress """
        for _ in range(n_estimators):
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity_split=self.min_impurity_split,
                max_depth=self.max_depth
            )
            self.trees.append(tree)

    def fit(self, X, y):
        """Initialize the y_pred
        example:
        [
            [ 0.30000001  0.32222223  0.37777779]
            [ 0.30000001  0.32222223  0.37777779]
            [ 0.30000001  0.32222223  0.37777779]
            ...
        ]
        """
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            """Very important step
            We use -(y - y_pred) to fit in the this tree.
            """
            gradient = self.loss_function.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            """Predict = learning_rate * update
            learning rate shrinks the contribution of each tree by learning_rate.
            """
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        if self._isClassifier:
            """Turn into probability distribution """
            y_pred = softmax(y_pred)
            """Select the label with maximum probability 
            y_pred = [[ 0.19578768  0.58784106  0.21637126]
            [ 0.5514167   0.22429165  0.22429165]
            [ 0.195787    0.21636848  0.58784452]
            ...
            ]

            np.argmax(y_pred, axis=1) = [1 0 2 ... ]
            """
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingClassifier(GradientBoosting):
    """Gradient Boosting for classification.
    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage n_classes_ regression trees are fit on the negative gradient
    of the binomial or multinomial deviance loss function.
    Binary classification is a special case where only a single regression tree is induced.
    """

    def __init__(self, n_estimators=100,
                 learning_rate=.1,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=3):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         min_samples_split=min_samples_split,
                                                         min_impurity_split=min_impurity_split,
                                                         max_depth=max_depth,
                                                         _isClassifier=True)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)


class GradientBoostingRegressor(GradientBoosting):
    """Gradient Boosting for regression.
    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the given loss function.
    """

    def __init__(self, n_estimators=100,
                 learning_rate=.1,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=3):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples_split=min_samples_split,
                                                        min_impurity_split=min_impurity_split,
                                                        max_depth=max_depth,
                                                        _isRegressor=True)
