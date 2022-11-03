from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from div import DivBoostClassifier


IRIS_PATH = "D:\\I sem ZSSI mgr\\Uczenie maszynowe\\Uczenie maszynowe projekt\\UCI-repository\\Iris\\"
base_estimator = DecisionTreeClassifier(max_depth=3)
n_estimators = 10
clfs = {
    'Ada': AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
    'Bag': BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
    "Div": DivBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
}
#letter-recognition
dataset = np.genfromtxt(IRIS_PATH+"letter-recognition.data", delimiter=",", dtype='str') #in case string type it prints nan
# dataset = np.genfromtxt(IRIS_PATH+"segment.data", delimiter=" ", dtype='str') #only for space delimeter STATLOG
switch = 0  #1 ONLY for not important 1st column
if switch == 0:
    try:
        X = dataset[:, :-1].astype(float)
        y = dataset[:, -1]  # if number type class at the beginning -> problem (we operate number class at the end)
    except:
        X = dataset[:, 1:].astype(float)
        y = dataset[:, 0]
elif switch == 1:
    X = dataset[:, 1:-1].astype(float)
    y = dataset[:, -1]  # if number type class at the beginning -> problem (we operate number class at the end)
    print(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.3,
    random_state=42
)

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(clfs), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clone(clfs[clf_name])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

np.save('results', scores)
scores = np.load('results.npy')
print("Folds:\n", scores)

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

headers = ["Ada", "Bag", "Div"]
names_column = np.array([["Ada"], ["Bag"], ["Div"]])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".3f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".3f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
(names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
(names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
(names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)