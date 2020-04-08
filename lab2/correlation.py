import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def correl_matrix(df, mean, std):
    cov = []
    xx = []
    xx_names = []
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i >= j:
                continue
            else:
                xx.append(np.mean([df[i].values[k] * df[j].values[k] for k in range(len(df[0].values))]))
                xx_names.append('x{}x{}'.format(i, j))
                cov.append((mean[i]*mean[j] - xx[i])/(std[i]*std[j]))
    print('cov', cov)
    print(xx_names, '\n', xx)

    A = df.corr(method='pearson')
    # B = df.corr(method='kendall')
    # C = df.corr(method='spearman')
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # print(A,'\n', B,'\n', C, '\n')
    print(np.linalg.det(A))
    print(eigenvalues)
    print(eigenvectors)


def normalize(df, mean, std):
    '''
    Function implements normalization of dataframe
    by formula z = (x - mean)/std
    More inf: scipy.stats.zscore
    '''
    names = df.columns.values
    dz_cols = [(df[i].values - mean[i])/std[i]  for i in range(len(df.columns))]
    #dz_cols = [stats.zscore(df[i].values, ddof=1) for i in range(len(df.columns))]
    zipped = list(zip(names, dz_cols))
    data = dict(zipped)
    load = pd.DataFrame(data)
    print(load)
    return load


def normal_test(df):
    '''
    Function display information about
    every column ditribution inside dataset.
    normaltest test used(info: scipy.stats.normaltest)
    '''
    names = ['Statistics', 'P-value', 'Result']
    stat, p, res = [], [], []

    for i in range(len(df.columns)):
        a, b = stats.normaltest(df[i].values)
        stat.append(a)
        p.append(b)
        if b < 0.05:  # 0.05 is alpha
            res.append('Rejected')
        else:
            res.append('Accepted')

    zipped = list(zip(names, [stat, p, res]))
    data = dict(zipped)
    load = pd.DataFrame(data)
    print('\n\nResult of Python normaltest')
    print(load)


def graphs(df):
    '''
    Function that implements graphic creation
    for datasets number of rows in picture is 2
    and there are 4 columns, results presented
    in form of hists
    '''
    plt.style.use('seaborn')
    for i in range(len(df.columns)):
        plt.subplot(2, 3, i+1)
        plt.hist(df[i].values, bins=20, ec='orange')
        plt.title('Column #{}'.format(i+1))
    plt.tight_layout()
    plt.show()


def stat_info(df):
    '''
    This function show basic information about
    every column in dataset. Information
    include Mean and Standard Deviation
    '''
    names = ['Mean', 'Standard Deviation']
    mean, std = [], []

    for i in range(len(df.columns)):
        mean.append(df[i].values.mean())
        std.append(df[i].values.std())

    zipped = list(zip(names, [mean, std]))
    data = dict(zipped)
    stats = pd.DataFrame(data)
    print(stats)
    return mean, std


if __name__ == "__main__":
    a = pd.read_csv('kpi17.txt', sep='\s+', header=None)
    df = a.loc[:, a.columns != 0]
    df.columns = [i for i in range(6)]
    graphs(df)
    mean, std = stat_info(df)
    # normal_test(df)
    # graphs(df)
    dz = normalize(df, mean, std)
    correl_matrix(dz, mean, std)
