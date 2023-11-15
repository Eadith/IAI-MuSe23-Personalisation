import re
# data = re.search("Personalising for (\d+).*Before personalisation .*CCC]: (.*)\n.*After personalisation \[CCC]:  (.*) \(",txt,re.S)
# match = re.search("Personalising for (\d+).*?\[CCC]: (.*?)\n.*?After personalisation.*?]:  (.*?) \(Difference",txt,re.S)

# result = []

# for part in txt.split("Find cached data")[-1:]:
# for part in txt.split("Saving data...")[1:]:
#     temp_re = []
#     for i in part.split("Difference")[:-1]:
#         Personalising = re.search("Personalising for (\d+)\n",i).group(1)
#         bval = re.search("Before personalisation.*]: +(-?0\.\d+)", i).group(1)
#         aval = re.search("After personalisation.*]: +(-?0\.\d+)",i).group(1)
#         temp_re.append((Personalising,bval,aval))
#     m = re.search("Results:\n\[Val]:  (.*?)\n",part.split("Difference")[-1])
#     temp_re.append(m.group(1))
#     result.append(temp_re)
# data = [
#     "egemaps-AROUSAL",
#     "ds-AROUSAL",
#     "w2v-AROUSAL",
#     "faus-AROUSAL",
#     "vit-AROUSAL",
#     "facenet-AROUSAL",
#     "egemaps-VALENCE",
#     "ds-VALENCE",
#     "w2v-VALENCE",
#     "faus-VALENCE",
#     "vit-VALENCE",
#     "facenet-VALENCE"
#  ]

# for i,d in zip(data,result):
#     print(fr"{i} : {d[-1]}")
#     print("p  before after")
#     for j in d[:-1]:
#         print(", ".join(j))
#     print("-"*15)
# print()



def one_sample(one_text):
    temp_re = []
    for i in one_text.split("Difference")[:-1]:
        Personalising = re.search("Personalising for (\d+)\n", i).group(1)
        bval = re.search("Before personalisation.*]: +(-?0\.\d+)", i).group(1)
        aval = re.search("After personalisation.*]: +(-?0\.\d+)", i).group(1)
        temp_re.append((Personalising, bval, aval))
    m = re.search("Results:\n\[Val]:  (.*?)\n", one_text.split("Difference")[-1])
    temp_re.append(m.group(1))
    print(temp_re[-1])
    print("p  before after")
    for j in temp_re[:-1]:
        print(", ".join(j))
    print("-"*15)

if __name__ == '__main__':
    txt = open("log.txt", "r").read()
    one_sample(txt)