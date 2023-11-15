import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main1():
    # data1 = pd.read_csv("./devel/apvit_v/predictions_test.csv")
    # data2 = pd.read_csv("./devel/ege_apvit_v/predictions_test.csv")
    # data1 = pd.read_csv("./devel/ds_aff_a/predictions_test.csv")
    # data2 = pd.read_csv("./devel/w2v_aff_a/predictions_test.csv")
    # data1 = pd.read_csv("./devel/ds_aff_a/predictions_devel.csv")
    # data2 = pd.read_csv("./devel/w2v_aff_a/predictions_devel.csv")
    data1 = pd.read_csv("./devel/apvit_v/predictions_devel.csv")
    data2 = pd.read_csv("./devel/ege_apvit_v/predictions_devel.csv")
    data = pd.concat([data1.iloc[:,1],data2.iloc[:,1]], axis=1)
    # data.columns = ["ap0.82","ege,apvit0.8347"]
    # data.columns = ["dsaff0.816","w2vaff0.81"]
    data.columns = ["apvit_v","ege_apvit_v"]
    data['mean'] = (data[data.columns[0]] + data[data.columns[1]]) / 2
    data['label'] = data1.iloc[:,2]

    temp = -1
    aa = []
    bb = []
    for i,id in enumerate(data1['meta_subj_id']):
        if id != temp:
            aa.append(i)
            bb.append(id)
        temp = id


    a = data.plot()
    plt.title("v_devel")
    plt.xticks(aa,bb)
    plt.show()
    print()


def main2():
    data1 = pd.read_csv("./devel/mvfacea_fx/predictions_devel.csv")
    temp = -1
    aa = []
    bb = []
    for i, id in enumerate(data1['meta_subj_id']):
        if id != temp:
            aa.append(i)
            bb.append(id)
        temp = id
    del data1['meta_subj_id']
    a = data1.plot()
    plt.title("a_mvface")
    plt.xticks(aa, bb)
    plt.show()
    print()


# a 17 54
# v 17 64  13 2
if __name__ == '__main__':
    main2()