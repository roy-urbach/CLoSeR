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


def classify_head_eval(dataset, m=lambda x: x.reshape(x.shape[0], -1), pca=False, linear=True, samples=0, k=10,
                       **kwargs):
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
    model.fit(train_embd, y_train.flatten()[inds])
    train_score = model.score(train_embd, y_train.flatten()[inds])
    test_score = model.score(m(x_test), y_test.flatten())
    print(
        f"Train acc: {train_score:.5f}; Test acc: {test_score:.5f}")
    return train_score, test_score
