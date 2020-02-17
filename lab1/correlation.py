import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def normal_test(df):
    '''
    Function display information about
    every column ditribution inside dataset.
    Shapiro-Wilk test used(info: scipy.stats.shapiro)
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
        plt.subplot(2, 4, i+1)
        plt.hist(df[i].values, bins=10, ec='orange')
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


if __name__ == "__main__":
    df = pd.read_csv('row_data.txt', sep='\s+', header=None)
    stat_info(df)
    normal_test(df)
    graphs(df)
