import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import mode
from math import factorial


class DivBoostClassifier(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        X, y = check_X_y(X, y)  # check shape of data
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        self.esemble_ = []

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=.3,
            random_state=42
        )

        N = len(X_train)
        scores = []
        w = [1/N for wj in range(N)]

        for k in range(self.n_estimators):
            self.esemble_.append(clone(self.base_estimator).fit(X_train, y_train, sample_weight = w))  # fit NOT on overall set, but on subset of D_trn
            y_pred = self.esemble_[k].predict(X_train)
            print(accuracy_score(y_train, y_pred))  # maybe train classifier at start, and next change weights
            scores.append(accuracy_score(y_train, y_pred))
            l = []
            e_k = 0
            for j in range(N):
                if y_pred[j] != y_train[j]:
                    l.append(1)
                    e_k+=w[j]
                else:
                    l.append(0)

            if e_k == 0 or e_k >= 0.5:
                w = [1/N for wj in range(N)]
            else:
                b_k = e_k/(1-e_k)
                denominator = 0
                for j in range(N):
                    for i in range(N):
                        denominator += w[i]*b_k**(1-l[j])
                    w[j] = w[j]*b_k**(1-l[j])/denominator
                    denominator = 0
        self.ced(self.esemble_, X_val, y_val)
        print("Hard voting - accuracy score: %.3f" % (np.mean(scores)))
        return self

    def cl_selection(self, C_f, k=1):
        print(C_f)
        index_C_f = np.argsort(C_f)[::-1][:k]
        print(index_C_f)
        return index_C_f

    def ced(self, L, X_val, y_val):
        S = []
        C_l = L.copy()
        C_ld = []
        lam = 0
        S.append(C_l[0])  # may change for random
        del C_l[0]
        N = len(L)
        while len(C_l) != 0:
            C_ld = []  # diversities of C_l classifiers
            for l_ind, l in enumerate(C_l):  # diversity distribution
                l_pred = l.predict(X_val)  # maybe in new "for" out of while
                fi_i = 0.0
                for j, S_j in enumerate(S):
                    fi_i += factorial(len(S))*factorial(N-len(S)-1)*cohen_kappa_score(l_pred, S_j.predict(X_val))  # is correct with def?
                fi_i /= (j+1)
                C_ld.append(fi_i)
            for a in self.cl_selection(C_ld):
                S.append(C_l[a])
                del C_l[a]
            if max(C_ld) < lam:
                print(max(C_ld))
                break
        print(S)
        return S

    def predict(self, X_test):
        predictions = []

        for clf in self.esemble_:
            predictions.append(clf.predict(X_test))

        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()