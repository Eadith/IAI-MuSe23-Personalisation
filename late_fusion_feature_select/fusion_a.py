import audmetric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def dev():
    data1 = pd.read_csv("devel/a_mvface_fx/predictions_devel.csv")
    data2 = pd.read_csv("devel/a_ds_aff/predictions_devel.csv")
    data3 = pd.read_csv("devel/a_w2v_aff/predictions_devel.csv")
    # data3_1 = pd.read_csv("devel/w2v_aff_a_new/predictions_devel.csv")
    data4 = pd.read_csv("devel/a_bert/predictions_devel.csv")
    # data5 = pd.read_csv("devel/a_mvface1/predictions_devel.csv")
    # data6 = pd.read_csv("devel/a_mvface2/predictions_devel.csv")

    weights = [0.5,0.25,0.25]

    pred = np.array([data1.iloc[:,1],
                     data2.iloc[:,1],
                     data3.iloc[:,1],
                     data4.iloc[:,1],
                     # data5.iloc[:,1],
                     # data6.iloc[:,1]
                     ]).mean(0)
    # pred_1 = np.array([data1.iloc[:,1],data2.iloc[:,1],data3_1.iloc[:,1],data4.iloc[:,1]]).mean(0)

    # p17 = np.array([data1.iloc[240:360,1]*weights[0],data2.iloc[240:360,1]*weights[1],data3.iloc[240:360,1]*weights[2]]).sum(0)
    # p54 = np.array([data1.iloc[1080:1200,1]*weights[0],data2.iloc[1080:1200,1]*weights[1],data3.iloc[1080:1200,1]*weights[2]]).sum(0)
    #
    # pred[240:360] = p17
    # pred[1080:1200] = p54


    label = data2.iloc[:,2]
    ccc1 = audmetric.concordance_cc(data1.iloc[:,1],label)
    print("ccc1",ccc1)
    ccc2 = audmetric.concordance_cc(data2.iloc[:,1], label)
    print("ccc2",ccc2)
    ccc3 = audmetric.concordance_cc(data3.iloc[:,1], label)
    print("ccc3",ccc3)
    ccc4 = audmetric.concordance_cc(data4.iloc[:,1], label)
    print("ccc4",ccc4)
    # ccc5 = audmetric.concordance_cc(data5.iloc[:,1], label)
    # print("ccc5",ccc5)
    # ccc6 = audmetric.concordance_cc(data6.iloc[:,1], label)
    # print("ccc6",ccc6)
    ccc = audmetric.concordance_cc(pred, label)
    print("ccc",ccc)

    pic_data = pd.DataFrame(np.concatenate([
                                            np.expand_dims(pred,1),
                                            # np.expand_dims(np.array(data4.iloc[:, 1]), 1),
                                            np.expand_dims(label,1),

                                            # np.expand_dims(pred_1,1)
                                            ],1))

    pic_data.columns = ["Mean", "label"]
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
    # plt.savefig("Mean.jpg",dpi=600)
    plt.show()

def test():
    data1 = pd.read_csv("devel/a_mvface_fx/predictions_test.csv")
    data2 = pd.read_csv("devel/a_ds_aff/predictions_test.csv")
    data3 = pd.read_csv("devel/a_w2v_aff/predictions_test.csv")
    # data3_1 = pd.read_csv("devel/w2v_aff_a_new/predictions_devel.csv")
    data4 = pd.read_csv("devel/a_bert/predictions_test.csv")
    data5 = pd.read_csv("devel/a_mvface1/predictions_test.csv")
    data6 = pd.read_csv("devel/a_mvface2/predictions_test.csv")
    weights = [0.5, 0.25, 0.25]

    pred = np.array([data1.iloc[:, 1],
                     data2.iloc[:, 1],
                     data3.iloc[:, 1],
                     data4.iloc[:, 1],
                     data5.iloc[:, 1],
                     data6.iloc[:, 1]
                     ]).mean(0)

    data4['pred'] = pred

    data4.to_csv("predictions_a.csv")
    print(audmetric.concordance_cc(data3.iloc[:, 1], pred))
    # p17 = np.array([data1.iloc[240:360,1]*weights[0],data2.iloc[240:360,1]*weights[1],data3.iloc[240:360,1]*weights[2]]).sum(0)
    # p54 = np.array([data1.iloc[1080:1200,1]*weights[0],data2.iloc[1080:1200,1]*weights[1],data3.iloc[1080:1200,1]*weights[2]]).sum(0)
    #
    # pred[240:360] = p17
    # pred[1080:1200] = p54


    pic_data = pd.DataFrame(np.concatenate([np.expand_dims(pred, 1),
                                            # np.expand_dims(np.array(data1.iloc[:, 1]), 1),
                                            np.expand_dims(np.array(data2.iloc[:, 1]), 1),
                                            # np.expand_dims(np.array(data3.iloc[:, 1]), 1),
                                            # np.expand_dims(np.array(data4.iloc[:, 1]), 1),
                                            ], 1))
    pic_data.columns = ["pred",
                        # "mvfacea_fx",
                        'ds_aff_a',
                        # 'w2v_aff_a',
                        # 'bert_a'
                        ]
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