# Provide statistical analysis of the obtained MAES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
from functools import reduce
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

def processData(pls=True, interaction=True, data_type='n', wine_type='red'):
    file_name = "data/" + wine_type + "_"
    if data_type == 'n':
        file_name += "normal"
    else:
        file_name += "data"
    file_name += ".csv"
    data = pd.read_csv(file_name)
    X, y = data.ix[:,:-1], data["quality"]
    # Optionally generate interaction features
    if interaction:
        old_features = [name for name in X.columns if not 'type' in name]
        for combination in combinations_with_replacement(old_features, 4):
            temp = reduce(lambda x, y: x*y, map(lambda x: X[x], combination))
            X.insert(X.shape[1]-1, '*'.join(combination), temp)
    # Normalize data
    X = (X - X.mean())/X.std()
    pca_flag = True
    interaction_flag = True
    # Optionally generate PCA components
    if pls:
        _pls = PLSRegression(n_components=30, scale=False)
        _pls.fit(X, y)
        X = pd.DataFrame(_pls.transform(X))
    return pd.concat([X, y], axis=1)

def getTriangulationScore(data):
    # Split the data
    train = data.sample(frac=0.8).sort_index()
    train_index = train.index
    test = data.loc[~data.index.isin(train_index)].copy(True)
    # Dictionary representing the gird. Each row represents a feature and the columns the partitions.
    # The shape of this matrix is a hyperparameter of our model.
    # The distance notion that we use is also a hyperparameter.
    correlations = pd.DataFrame(train.corr()['quality']).apply(np.abs)
    temp = correlations.sort_values(by='quality', ascending=False)[1:4]
    train.rename(columns={temp.index[0]:'split_a',temp.index[1]:'split_b', temp.index[2]: 'split_c' }, inplace=True)
    test.rename(columns={temp.index[0]:'split_a',temp.index[1]:'split_b', temp.index[2]: 'split_c' }, inplace=True)
    
    grid = {}
    for name in ['split_'+ char for char in ['a', 'b', 'c']]: #train.columns:
        grid[name] = []
    # For each feature we just give one separation point.
    for element in grid:
        grid[element] = np.median(train[element])
    grid
    # Data partitioning
    dfs = []
    dfs.append(train[(train.split_a > grid["split_a"]) 
                     & (train.split_b > grid["split_b"]) 
                     & (train.split_c > grid["split_c"])])
    dfs.append(train[(train.split_a > grid["split_a"]) 
                     & (train.split_b > grid["split_b"] ) 
                     & (train.split_c < grid["split_c"])])
    dfs.append(train[(train.split_a > grid["split_a"]) 
                     & (train.split_b < grid["split_b"]) 
                     & (train.split_c > grid["split_c"])])
    dfs.append(train[(train.split_a > grid["split_a"]) 
                     & (train.split_b < grid["split_b"]) 
                     & (train.split_c < grid["split_c"])])
    dfs.append(train[(train.split_a < grid["split_a"])
                     & (train.split_b > grid["split_b"]) 
                     & (train.split_c > grid["split_c"])])
    dfs.append(train[(train.split_a < grid["split_a"]) 
                     & (train.split_b > grid["split_b"])
                     & (train.split_c < grid["split_c"])])
    dfs.append(train[(train.split_a < grid["split_a"]) 
                     & (train.split_b < grid["split_b"]) 
                     & (train.split_c > grid["split_c"])])
    dfs.append(train[(train.split_a < grid["split_a"]) 
                     & (train.split_b < grid["split_b"]) 
                     & (train.split_c < grid["split_c"])])
    # Training and testing
    def trainLG(df):
        X, y = df.ix[:,:-1], df['quality']
        clf = LogisticRegression(penalty='l2', C=10, n_jobs=-1)
        clf.fit(X, y)
        return clf
    clfs = []
    for df in dfs:
        clfs.append(trainLG(df))

    X_test, y_test = test.ix[:,:-1], test['quality']

    def getPiece(row): 
        if row.split_a > grid["split_a"]:
            if row.split_b > grid["split_b"]:
                if row.split_c > grid["split_c"]:
                    return 0
                else:
                    return 1
            else:
                if row.split_c > grid["split_c"]:
                    return 2
                else:
                    return 3
        else:
            if row.split_b > grid["split_b"]:
                if row.split_c > grid["split_c"]:
                    return 4
                else:
                    return 5
            else:
                if row.split_c > grid["split_c"]:
                    return 6
                else:
                    return 7
            
    def generatePrediction(df): # predicts using the classifier from the region
        y_pred = []
        for idx in df.index:
            obs  = df.loc[idx]
            temp = getPiece(obs)
            jdx = clfs[temp].predict(obs.values.reshape(1,-1))
            y_pred.append(jdx[0])
        return pd.Series(y_pred)

    y_pred = generatePrediction(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    #print("MAE = {}\tAcc. = {}".format(mae, acc))
    return mae, acc


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        wine_type = sys.argv[1]
        num_trials = int(sys.argv[2])
        pls, interaction = sys.argv[3:-1]
        data_type = sys.argv[-1]
    else:
        wine_type = 'red'
        num_trials = 30
        pls, interaction = True, True
        data_type = 'n'

    data = processData(pls=pls,
                       interaction=interaction,
                       data_type=data_type,
                       wine_type=wine_type)
    
    maes = []
    accs = []
    for _ in range(num_trials):
        mae, acc = getTriangulationScore(data)
        maes.append(mae)
        accs.append(acc)
    maes = np.array(maes)
    accs = np.array(accs)

    print("Experimental results with {} trials:".format(num_trials))
    mean, std = maes.mean(), maes.std()
    print("mean = {}\nstd = {}".format(mean, std))
    error = (1.96 * std) / (num_trials**0.5)
    print("95% CI = [{}, {}]".format(mean - error, mean + error))
    maes = pd.DataFrame(maes)
    sns.distplot(maes)
    plt.title("Distribution of MAEs")
    plt.xlabel("mean = {:.3}".format(mean) + 10*" " + "std={:.3}".format(std))
    plt.show()
