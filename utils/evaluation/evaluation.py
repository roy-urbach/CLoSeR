import numpy as np
from sklearn.decomposition import PCA

from utils.data import Data
from utils.evaluation.ensemble import EnsembleModel, EnsembleVotingMethods
from utils.evaluation.utils import CS
from utils.utils import printd


def linear_head_eval(svm=True, C=1e-2, categorical=False, **kwargs):
    if categorical:
        if svm:
            from sklearn.svm import SVC
            model = SVC(kernel='linear', C=C, max_iter=int(1e7), **kwargs)
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=C if C > 0 else 1, penalty=None if C==0 else 'l2', **kwargs)
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=C, **kwargs)
    return model


def classify_head_eval(dataset, m=lambda x: x.reshape(x.shape[0], -1), categorical=False,
                       pca=False, linear=True, samples=0, k=10, CS=CS, all_CS=True, **kwargs):
    (x_train, y_train), (x_test, y_test) = dataset.get_all()
    if dataset.get_x_val() is not None and linear:
        val_dataset = Data(x_train, y_train, dataset.get_x_val(), dataset.get_y_val())
        best_score = None
        winner_C = None
        prev_train, prev_val = None, None
        for C in CS:
            print(f"{C=}")
            cur_train, cur_val = classify_head_eval(val_dataset, m=m, categorical=categorical, pca=pca, linear=linear,
                                                    samples=samples, k=k, C=C, **kwargs)
            if not all_CS and prev_train is not None and cur_train > prev_train and cur_val < prev_val:
                break
            if best_score is None or cur_val > best_score:
                winner_C = C
                best_score = cur_val
            prev_train = cur_train
            prev_val = cur_val
        kwargs.update(C=winner_C)

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
        model = linear_head_eval(categorical=categorical, **kwargs)
    else:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if categorical:
            model = KNeighborsClassifier(k)
        else:
            model = KNeighborsRegressor(k, weights='distance')
    model.fit(train_embd, y_train[inds])
    train_score = model.score(train_embd, y_train[inds], **score_kwargs)
    test_score = model.score(m(x_test), y_test, **score_kwargs)
    if dataset.get_x_val() is not None:
        val_score = model.score(m(dataset.get_x_val()), dataset.get_y_val(), **score_kwargs)
    else:
        val_score = None
    print(
        f"Train acc: {train_score:.5f}; Test acc: {test_score:.5f}" + (f"; Val score: {val_score:.5f}" if val_score is not None else ""))
    return (train_score, test_score) if val_score is None else (train_score, val_score, test_score)


def classify_head_eval_ensemble(dataset, linear=True, k=10, base_name='',
                                categorical=False,
                                voting_methods=EnsembleVotingMethods, C=1, **kwargs):
    (x_train, y_train), (x_test, y_test) = dataset.get_all()
    if linear:
        model = linear_head_eval(C=C, categorical=categorical, **kwargs)
        name = 'linear'
    else:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if categorical:
            model = KNeighborsClassifier(k)
        else:
            model = KNeighborsRegressor(k, weights='distance')
        name = f'knn{k}'
    ensemble = EnsembleModel(model, get_score_method='predict_proba' if categorical else 'predict', ensemble_axis=-1)

    res = {}

    if dataset.get_x_val() is not None and linear:
        for i, voting_method in enumerate(voting_methods):
            (ind_train, ind_val, ind_test), (ens_train, ens_val, ens_test) = ensemble.fit_with_validation(x_train, y_train,
                                                                                                          dataset.get_x_val(), dataset.get_y_val(),
                                                                                                          x_test, y_test,
                                                                                                          voting_method=voting_method)
            if not i:
                for p, pathway in enumerate(ensemble.models):
                    res[base_name + f'pathway{p}_{name}'] = (ind_train[p], ind_val[p], ind_test[p])
                res[base_name + f'pathways_mean_{name}'] = np.mean(list(res.values()), axis=0).tolist()
            res[base_name + f"ensemble_{name}_" + voting_method.name] = (ens_train, ens_val, ens_test)
            printd(f"{base_name + voting_method.name} Train acc: {ens_train:.5f}; Val acc: {ens_val:.5f}; Test acc: {ens_test:.5f}")
    else:
        ensemble.fit(x_train, y_train.flatten())

        for i, pathway in enumerate(ensemble.models):
            from utils.data import Data
            cur_ds = Data(x_train[..., i], y_train, x_test[..., i], y_test)
            path_score_train = pathway.score(*cur_ds.get_train())
            path_score_test = pathway.score(*cur_ds.get_test())
            path_scores = [path_score_train, path_score_test]
            if dataset.get_x_val() is not None:
                path_score_val = pathway.score(dataset.get_x_val()[..., i], dataset.get_y_val())
                path_scores = (path_score_train, path_score_val, path_score_test)
            res[base_name + f'pathway{i}_{name}'] = path_scores

        res[base_name + f'pathways_mean_{name}'] = np.mean(list(res.values()), axis=0).tolist()

        for voting_method in voting_methods:
            ens_train_score = ensemble.score(x_train, y_train, voting_method=voting_method)
            ens_test_score = ensemble.score(x_test, y_test, voting_method=voting_method)
            ens_scores = (ens_train_score, ens_test_score)

            if dataset.get_x_val() is not None:
                ens_val_score = ensemble.score(dataset.get_x_val(), dataset.get_y_val(), voting_method=voting_method)
                ens_scores = (ens_train_score, ens_val_score, ens_test_score)
            else:
                ens_val_score = None

            res[base_name + f"ensemble_{name}_" + voting_method.name] = ens_scores
            printd(f"{base_name + voting_method.name} Train acc: {ens_train_score:.5f}; Test acc: {ens_test_score:.5f}" + (f"; Val acc: {ens_val_score}" if ens_val_score is not None else ""))
    return res
