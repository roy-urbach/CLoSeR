import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def linear_head_eval(svm=True):
    if svm:
        from sklearn.svm import SVC
        model = SVC(kernel='linear', max_iter=int(1e7))
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    return model


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
                       for X_train_single in self.split(X_train)]

    def predict(self, X):
        res = np.sum([getattr(model, self.get_score)(X_single)
                      for model, X_single in zip(self.models, self.split(X))], axis=0)
        return np.argmax(res, axis=-1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def classify_head_eval(dataset, m=lambda x: x.reshape(x.shape[0], -1), pca=False, linear=True, samples=0, k=10,
                       ensemble=False, **kwargs):
    (x_train, y_train), (x_test, y_test) = dataset.get_all()

    if pca:
        pca = PCA(pca).fit(x_train.reshape(x_train.shape[0], -1))
        m = lambda x: pca.transform(x.reshape(x.shape[0], -1))
    if samples:
        inds = np.random.permutation(len(x_train))[:samples]
    else:
        inds = np.arange(len(x_train))
    train_embd = m(x_train[inds])
    if linear:
        model = linear_head_eval(**kwargs)
    else:
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(k)
    if ensemble:
        model = EnsembleModel(model, ensemble_axis=1)
    model.fit(train_embd, y_train.flatten()[inds])
    train_score = model.score(train_embd, y_train.flatten()[inds])
    test_score = model.score(m(x_test), y_test.flatten())
    print(
        f"Train acc: {train_score:.5f}; Test acc: {test_score:.5f}")
    return train_score, test_score
