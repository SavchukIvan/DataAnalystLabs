import numpy as np
import pandas as pd


def twoway_anova(df):
    dff = prep(df)
    a = []
    b = []
    for i in range(len(dff.columns)):
        a.append(sum(dff[i].values))
    for i in range(len(dff[0].values)):
        b.append(sum(dff.loc[i, :]))
    print(sum(a), sum(b))


def prep(df):
    sl = int(len(df[1].values)/4)
    dff = []
    for i in range (len(df.columns)):
        dff.append(df[i].values)
    dff1 = []
    dff2 = []
    dff3 = []
    dff4 = []
    dffn = [dff1, dff2, dff3, dff4]
    for j in range (len(df.columns)):
        dff1.append(dff[j][:sl])
        dff2.append(dff[j][sl:(2*sl)])
        dff3.append(dff[j][(2*sl):(3 * sl)])
        dff4.append(dff[j][(3*sl):(4 * sl)])
    a = []
    for i in range(len(dffn)):
        b = [np.mean(dffn[i][j]) for j in range(len(dffn[i]))]
        a.append(b)
    names = [0, 1, 2, 3]
    zipped = list(zip(names, a))
    data = dict(zipped)
    stats_t = pd.DataFrame(data)
    stats = stats_t.transpose()
    return stats
    

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
    print(stats)

    g = max(varc) / sum(varc)
    print(g)
    


if __name__ == "__main__":
    a = pd.read_csv('kpi17.txt', sep='\s+', header=None)
    df = a.loc[:, a.columns != 0]
    df.columns = [i for i in range(6)]
    #variance(df)
    #print(df.head(4))
    #anova(df)
    twoway_anova(df)
