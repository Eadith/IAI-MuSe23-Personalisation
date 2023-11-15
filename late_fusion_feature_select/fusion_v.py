import audmetric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dev():
    data1 = pd.read_csv("devel/v_bert_1/predictions_devel.csv")
    data2 = pd.read_csv("devel/v_bert_2/predictions_devel.csv")
    data3 = pd.read_csv("devel/v_ege_apvit/predictions_devel.csv")
    data4 = pd.read_csv("devel/v_apvit/predictions_devel.csv")
    data5 = pd.read_csv("devel/v_mvface_1/predictions_devel.csv")
    data6 = pd.read_csv("devel/v_mvface_2/predictions_devel.csv")
    # data7 = pd.read_csv("devel/v_mvface1/predictions_devel.csv")
    # data8 = pd.read_csv("devel/v_mvface2/predictions_devel.csv")
    # data9 = pd.read_csv("devel/v_affect71/predictions_devel.csv")
    pred_ = np.array([data1.iloc[:,1],data2.iloc[:,1]]).mean(0)

    pred = np.array([data3.iloc[:,1],
                     data4.iloc[:,1],
                     data5.iloc[:,1],
                     data6.iloc[:,1],
                     # data7.iloc[:, 1],
                     # data9.iloc[:, 1],
                     ]).mean(0)

    # pred[1200:1320] = np.array([pred[1200:1320],data9.iloc[1200:1320, 1]]).mean(0)

    pred[360:480] = pred_[360:480] # 2
    # pred[240:360] = np.array([data5.iloc[:, 1], data6.iloc[:, 1]]).mean(0)[240:360] # 17
    # pred = np.array([data1.iloc[:,1],data2.iloc[:,1],data3.iloc[:,1],data4.iloc[:,1]]).mean(0)
    # pred = np.array([data1.iloc[:,1],data2.iloc[:,1],data3.iloc[:,1]]).mean(0)
    # pred = np.array([data1.iloc[:,1],data2.iloc[:,1]]).mean(0)

    # p17 = np.array([data1.iloc[240:360,1]*weights[0],data2.iloc[240:360,1]*weights[1],data3.iloc[240:360,1]*weights[2]]).sum(0)
    # p54 = np.array([data1.iloc[1080:1200,1]*weights[0],data2.iloc[1080:1200,1]*weights[1],data3.iloc[1080:1200,1]*weights[2]]).sum(0)
    #
    # pred[240:360] = p17
    # pred[1080:1200] = p54

    # print(audmetric.concordance_cc(data4.iloc[:, 1], pred))

    '''
                        6       5       4       3       2       1       7        8
    17 [240:360]     0.24      0.18    0.09    0.16   0.01    0.20     0.186    0.125   0.17
    56 [1200:1320]   0.06      0.02    0.04    0.007  0.045   0.09     0.02     0.01   0.04
    64 [1440:1560]   0.185     0.39    0.54    0.65   0.51    0.26    0.53      0.31    0.57
    '''
    label = data2.iloc[:,2]
    ccc1 = audmetric.concordance_cc(data1.iloc[:,1],label)
    print("ccc1",ccc1)
    ccc2 = audmetric.concordance_cc(data2.iloc[:,1], label)
    print("ccc2",ccc2)
    ccc3 = audmetric.concordance_cc(data3.iloc[:,1], label)
    print("ccc3",ccc3)
    ccc4 = audmetric.concordance_cc(data4.iloc[:,1], label)
    print("ccc4",ccc4)
    ccc5 = audmetric.concordance_cc(data5.iloc[:,1], label)
    print("ccc5",ccc5)
    ccc6 = audmetric.concordance_cc(data6.iloc[:,1], label)
    print("ccc6",ccc6)
    # ccc7 = audmetric.concordance_cc(data7.iloc[:,1], label)
    # print("ccc7",ccc7)
    # ccc8 = audmetric.concordance_cc(data8.iloc[:,1], label)
    # print("ccc8",ccc8)
    # ccc9 = audmetric.concordance_cc(data9.iloc[:,1], label)
    # print("ccc9",ccc9)
    ccc = audmetric.concordance_cc(pred, label)
    print("ccc",ccc)


    pic_data = pd.DataFrame(np.concatenate([
                                        np.expand_dims(pred,1),
                                        # np.expand_dims(np.array(data4.iloc[:, 1]), 1),
                                        np.expand_dims(label,1),
                                            ],1))
    pic_data.columns = ["Mean","label"]
    pic_data.plot()
    aa = []
    bb = []
    temp = -1
    for i, id in enumerate(data1['meta_subj_id']):
        if id != temp:
            aa.append(i)
            bb.append(id)
        temp = id

    plt.xticks(aa, bb)
    plt.savefig("Mean.jpg", dpi=600)
    plt.show()


def test():
    data1 = pd.read_csv("devel/v_bert_1/predictions_test.csv")
    data2 = pd.read_csv("devel/v_bert_2/predictions_test.csv")
    data3 = pd.read_csv("devel/v_ege_apvit/predictions_test.csv")
    data4 = pd.read_csv("devel/v_apvit/predictions_test.csv")
    data5 = pd.read_csv("devel/v_mvface_1/predictions_test.csv")
    data6 = pd.read_csv("devel/v_mvface_2/predictions_test.csv")
    data7 = pd.read_csv("devel/v_mvface1/predictions_test.csv")
    data8 = pd.read_csv("devel/v_mvface2/predictions_test.csv")
    data9 = pd.read_csv("devel/v_affect71/predictions_test.csv")


    pred_ = np.array([data1.iloc[:,1],data2.iloc[:,1]]).mean(0)

    pred = np.array([data3.iloc[:,1],
                     data4.iloc[:,1],
                     data5.iloc[:,1],
                     data6.iloc[:,1],
                     data7.iloc[:,1]
                     ]).mean(0)

    pred[3684:4038] = np.array([pred[3684:4038],data9.iloc[3684:4038, 1]]).mean(0) # 56

    pred[1006:1372] = pred_[1006:1372] # 2
    data4['pred'] = pred
    data4.to_csv("predictions_v.csv")
    # pred = np.array([data1.iloc[:,1],data2.iloc[:,1],data3.iloc[:,1],data4.iloc[:,1]]).mean(0)
    # pred = np.array([data1.iloc[:,1],data2.iloc[:,1],data3.iloc[:,1]]).mean(0)
    # pred = np.array([data1.iloc[:,1],data2.iloc[:,1]]).mean(0)

    # p17 = np.array([data1.iloc[240:360,1]*weights[0],data2.iloc[240:360,1]*weights[1],data3.iloc[240:360,1]*weights[2]]).sum(0)
    # p54 = np.array([data1.iloc[1080:1200,1]*weights[0],data2.iloc[1080:1200,1]*weights[1],data3.iloc[1080:1200,1]*weights[2]]).sum(0)
    #
    # pred[240:360] = p17
    # pred[1080:1200] = p54

    print(audmetric.concordance_cc(data4.iloc[:, 1], pred))


    label = data2.iloc[:,2]
    ccc1 = audmetric.concordance_cc(data1.iloc[:,1],label)
    print("ccc1",ccc1)
    ccc2 = audmetric.concordance_cc(data2.iloc[:,1], label)
    print("ccc2",ccc2)
    ccc3 = audmetric.concordance_cc(data3.iloc[:,1], label)
    print("ccc3",ccc3)
    ccc4 = audmetric.concordance_cc(data4.iloc[:,1], label)
    print("ccc4",ccc4)
    ccc5 = audmetric.concordance_cc(data5.iloc[:,1], label)
    print("ccc5",ccc5)
    ccc6 = audmetric.concordance_cc(data6.iloc[:,1], label)
    print("ccc6",ccc6)
    ccc = audmetric.concordance_cc(pred, label)
    print("ccc",ccc)



    pic_data = pd.DataFrame(np.concatenate([np.expand_dims(pred,1),np.expand_dims(label,1),
                                            ],1))
    pic_data.columns = ["pred","label",]
    pic_data.plot()
    aa = []
    bb = []
    temp = -1
    for i, id in enumerate(data1['meta_subj_id']):
        if id != temp:
            aa.append(i)
            bb.append(id)
        temp = id

    plt.xticks(aa, bb)
    plt.show()


if __name__ == '__main__':
    dev()