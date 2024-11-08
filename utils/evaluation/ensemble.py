from enum import Enum
import numpy as np
from functools import partial

from utils.data import Data
from utils.evaluation.utils import CS
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
    def __init__(self, model, get_score_method="predict_proba", ensemble_axis=0, CS=CS):
        self.base_model = model
        self.get_score = get_score_method
        assert hasattr(self.base_model, get_score_method)
        self.ensemble_axis = ensemble_axis
        self.models = []
        self.CS = CS
        self.CS_models = []
        self.best_CS = None
        self.scores_val = None
        self.scores_train = None
        self.scores_test = None
        self.best_CS = None

    def split(self, X):
        return [X_single.squeeze(axis=self.ensemble_axis)
                for X_single in np.split(X, X.shape[self.ensemble_axis], axis=self.ensemble_axis)]

    def fit_with_validation(self, X_train, y_train, X_val, y_val, X_test, y_test, voting_method=EnsembleVotingMethods.ArgmaxMeanProb, individual_ys=False):
        if not self.CS_models:
            self.scores_val = [None] * X_train.shape[self.ensemble_axis]
            self.scores_train = [None] * len(self.scores_val)
            self.scores_test = [None] * len(self.scores_val)
            self.best_CS = [None] * len(self.scores_val)
            for C in self.CS:
                print(f"{C=}")
                if C == 0:
                    self.base_model.penalty = None
                    self.base_model.C = 1
                else:
                    self.base_model.penalty = 'l2'
                    self.base_model.C = C
                self.fit(X_train, y_train, CS=[C]*len(self.scores_val))
                self.CS_models.append(self.models)
                for i, model in enumerate(self.models):
                    model_val_score = model.score(np.take(X_val, i, axis=self.ensemble_axis), y_val[..., i] if individual_ys else y_val)
                    if self.scores_val[i] is None or model_val_score > self.scores_val[i]:
                        self.scores_val[i] = model_val_score
                        self.best_CS[i] = C
                        self.scores_test[i] = model.score(np.take(X_test, i, axis=self.ensemble_axis), y_test[..., i] if individual_ys else y_test)
                        self.scores_train[i] = model.score(np.take(X_train, i, axis=self.ensemble_axis), y_train[..., i] if individual_ys else y_train)
            self.models = [self.CS_models[np.where(self.CS == C)[0][0]][i] for i, C in enumerate(self.best_CS)]

        ensemble_score_train = None
        ensemble_score_val = None
        ensemble_score_test = None

        if voting_method is not None:

            for C in self.CS:
                ens_score = self.score(X_val, y_val, CS=[C]*len(self.models), voting_method=voting_method)
                if ensemble_score_val is None or ens_score > ensemble_score_val:
                    ensemble_score_val = ens_score
                    ensemble_score_test = self.score(X_test, y_test, voting_method=voting_method, CS=CS)
                    ensemble_score_train = self.score(X_train, y_train, voting_method=voting_method, CS=CS)

            if np.unique(self.best_CS).size > 1:
                ens_score = self.score(X_val, y_val, voting_method=voting_method)
                if ensemble_score_val is None or ens_score > ensemble_score_val:
                    print("combination of CS got best ensemble results...")
                    ensemble_score_val = ens_score
                    ensemble_score_train = self.score(X_train, y_train, voting_method=voting_method)
                    ensemble_score_test = self.score(X_test, y_test, voting_method=voting_method)

        return (self.scores_train, self.scores_val, self.scores_test), (ensemble_score_train, ensemble_score_val, ensemble_score_test)

    def fit(self, X_train, y_train, CS=None, individual_ys=False):
        self.models = []
        from sklearn.base import clone as sklearn_clone
        for i, X_train_single in counter(enumerate(self.split(X_train))):
            model = sklearn_clone(self.base_model)
            if CS is not None:
                C = CS[i]
                if C == 0:
                    self.base_model.penalty = None
                    self.base_model.C = 1
                else:
                    self.base_model.penalty = 'l2'
                    self.base_model.C = C
            self.models.append(model.fit(X_train_single, y_train[..., i] if individual_ys else y_train))

    def predict(self, X, voting_method=EnsembleVotingMethods.ArgmaxMeanProb, CS=None):
        models = self.models if CS is None else [self.CS_models[np.where(self.CS == C)[0][0]][i] for i, C in enumerate(CS)]
        probs = np.stack([getattr(model, self.get_score)(X_single)
                          for model, X_single in zip(models, self.split(X))], axis=-1)
        return voting_method.value(probs)

    def score(self, X, y, voting_method=EnsembleVotingMethods.ArgmaxMeanProb, CS=None):
        return np.mean(self.predict(X, voting_method=voting_method, CS=CS) == y)
