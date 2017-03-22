import pandas as pd
import numpy as np

from itertools import combinations_with_replacement, product
from functools import reduce

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix


def process_data(X, degree, n_components, scale):
    old_features = X.columns
    if degree > 1:
        for combination in combinations_with_replacement(old_features, degree):
            temp = reduce(lambda x, y: x*y, map(lambda x: X[x], combination))
            X.insert(X.shape[1]-1, '*'.join(combination), temp)
    X = (X - X.mean())/X.std()
    pls = PLSRegression(n_components=n_components, scale=scale)
    pls.fit(X, y)
    X = pd.DataFrame(pls.transform(X))

def my_train_score(X_train, X_test, y_train, y_test,
                   penalty, C, class_weight, solver,multi_class):
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.set_params(penalty=penalty,
                   C=C,
                   class_weight=class_weight,
                   solver=solver,
                   multi_class=multi_class)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def legal_param(param, X):
    n_components = param[1]
    penalty = param[3]
    solver = param[6]
    multi_class = param[7]
    if n_components > X.shape[1]:
        return False
    if solver in ['newton-cg', 'lbfgs', 'sag']:
         return penalty == 'l2'
    else:
        return multi_class == 'ovr' 
    

if __name__ == "__main__":
    import sys
    #load data
    if len(sys.argv) == 1:
        data = pd.read_csv("data/red_normal.csv")
        #data = pd.read_csv("data/red_data.csv")
    else:
        data = pd.read_csv("data/{}_normal.csv".format(sys.argv[1]))
        #data = pd.read_csv("data/{}_data.csv".format(sys.argv[1]))

    train_index = data.sample(frac=0.8).sort_index().index
    test_index = ~data.index.isin(train_index)
    
    # ranges of parameters for exploration
    degrees = range(1, 6)
    n_components_range = range(5, 50, 5)
    scales = [True, False]
    penalties = ['l1', 'l2']
    C_range = [0.0001*10**i for i in range(7)]
    class_weights = ['balanced', None]
    solvers = ['newton-cg', 'lbfgs', 'sag', 'liblinear']
    multi_classes = ['multinomial', 'ovr']

    my_iterator = product(degrees, n_components_range, scales, penalties,
                          C_range, class_weights, solvers, multi_classes)
    maes = list()
    params = list()
    for param in my_iterator:
        X, y = data.ix[:,:-1], data["quality"]
        degree, n_components, scale, penalty, C, class_weight, solver, multi_class = param
        if not legal_param(param, X):
            continue
        params.append(param)

        process_data(X, degree, n_components, scale)
        
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        mae = my_train_score(X_train, X_test, y_train, y_test,
                             penalty, C, class_weight, solver,multi_class)
        maes.append(mae)
        
    arg_min = np.argmin(maes)
    print("Min MAE={}".format(maes[arg_min]))
    print(params[arg_min])
