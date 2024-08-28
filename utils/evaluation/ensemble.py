from enum import Enum
import numpy as np
from functools import partial
from utils.utils import counter


class EnsembleVotingMethods(Enum):
    """
    Assumes shape (samples, classes, pathways)
    """
    ArgmaxMeanProb = partial(lambda probs: np.argmax(probs.mean(axis=-1), axis=1))
    ArgmaxMaxProb = partial(lambda probs: np.argmax(probs.max(axis=-1), axis=1))
    ArgmaxMeanLogProb = partial(lambda probs: np.argmax(np.log2(probs).mean(axis=-1), axis=1))
    MajorityVote = partial(lambda probs: np.array([np.argmax(np.bincount(s)) for s in np.argmax(probs, axis=1)]))
    Mean = partial(lambda preds: np.mean(preds, axis=1))


class EnsembleModel:
    def __init__(self, model, get_score_method="predict_proba", ensemble_axis=0):
        self.base_model = model
        self.get_score = get_score_method
        assert hasattr(self.base_model, get_score_method)
        self.ensemble_axis = ensemble_axis
        self.models = []

    def split(self, X):
        return [X_single.squeeze(axis=self.ensemble_axis)
                for X_single in np.split(X, X.shape[self.ensemble_axis], axis=self.ensemble_axis)]

    def fit(self, X_train, y_train):
        from sklearn.base import clone as sklearn_clone
        self.models = [sklearn_clone(self.base_model).fit(X_train_single, y_train)
                       for X_train_single in counter(self.split(X_train))]

    def predict(self, X, voting_method=EnsembleVotingMethods.ArgmaxMeanProb):
        probs = np.stack([getattr(model, self.get_score)(X_single)
                          for model, X_single in zip(self.models, self.split(X))], axis=-1)
        return voting_method.value(probs)

    def score(self, X, y, voting_method=EnsembleVotingMethods.ArgmaxMeanProb):
        return np.mean(self.predict(X, voting_method=voting_method) == y)
