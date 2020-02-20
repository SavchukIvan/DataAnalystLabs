import numpy as np
import pandas as pd


def anova(df):
    s_per_col = []
    q1 = 0
    for i in range(len(df.columns)):
        s_per_col.append(sum(df[i].values))
        q1 += np.square((df[i].values)).sum()
    q2 = 1/len(df[1].values) * sum(np.square(s_per_col))
    q3 = 1/(len(df[1].values) * len(df.columns)) * pow(sum(s_per_col), 2)
    so_sq = (q1 - q2)/((len(df[1].values) - 1)*len(df.columns))
    sa_sq = (q2 - q3)/(len(df.columns) - 1)
    fract = sa_sq / so_sq
    f1 = len(df.columns) - 1
    f2 = len(df.columns)* (len(df[1].values) - 1)
    f_fish = 2.21
    if fract > f_fish:
        print('Factor is significant')
    else:
        print('Factor is not significant')


def variance(df):
    varc = []
    name = ['Varince']
    for i in range(len(df.columns)):
        varc.append(df[i].values.var())
    
    zipped = list(zip(name, [varc]))
    data = dict(zipped)
    stats = pd.DataFrame(data)
    #print(stats)

    g = max(varc) / sum(varc)
    print(g)
    


if __name__ == "__main__":
    a = pd.read_csv('kpi17.txt', sep='\s+', header=None)
    df = a.loc[:, a.columns != 0]
    df.columns = [i for i in range(6)]
    print(df.head(4))
    anova(df)
