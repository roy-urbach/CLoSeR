import numpy as np
from sklearn.decomposition import PCA

from evaluation.ensemble import EnsembleModel, EnsembleVotingMethods
from utils.utils import printd


def linear_head_eval(svm=True, C=1e-2, **kwargs):
    if svm:
        from sklearn.svm import SVC
        model = SVC(kernel='linear', C=C, max_iter=int(1e7), **kwargs)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=C, **kwargs)
    return model


def classify_head_eval(dataset, m=lambda x: x.reshape(x.shape[0], -1), pca=False, linear=True, samples=0, k=10, **kwargs):
    (x_train, y_train), (x_test, y_test) = dataset.get_all()
    score_kwargs = {}

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
    model.fit(train_embd, y_train[inds])
    train_score = model.score(train_embd, y_train[inds], **score_kwargs)
    test_score = model.score(m(x_test), y_test, **score_kwargs)
    print(
        f"Train acc: {train_score:.5f}; Test acc: {test_score:.5f}")
    return train_score, test_score


def classify_head_eval_ensemble(dataset, linear=True, k=10, base_name='',
                                voting_methods=EnsembleVotingMethods, C=1, **kwargs):
    (x_train, y_train), (x_test, y_test) = dataset.get_all()
    if linear:
        model = linear_head_eval(C=C, **kwargs)
        name = 'linear'
    else:
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(k)
        name = f'knn{k}'
    ensemble = EnsembleModel(model, ensemble_axis=-1)
    ensemble.fit(x_train, y_train.flatten())

    res = {}

    for i, pathway in enumerate(ensemble.models):
        from utils.data import Data
        cur_ds = Data(x_train[..., i], y_train, x_test[..., i], y_test)
        res[base_name + f'pathway{i}_{name}'] = (pathway.score(*cur_ds.get_train()),
                                                 pathway.score(*cur_ds.get_test()))
    res[base_name + f'pathways_mean_{name}'] = np.mean(list(res.values()), axis=0).tolist()

    for voting_method in voting_methods:
        train_score = ensemble.score(x_train, y_train, voting_method=voting_method)
        test_score = ensemble.score(x_test, y_test, voting_method=voting_method)
        res[base_name + f"ensemble_{name}_" + voting_method.name] = (train_score, test_score)
        printd(f"{base_name + voting_method.name} Train acc: {train_score:.5f}; Test acc: {test_score:.5f}")
    return res
