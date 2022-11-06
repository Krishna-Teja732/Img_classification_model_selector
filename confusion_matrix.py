import pickle
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

const_header = {
    0: "bachoe loader",
    1: "bricks",
    2: "cement",
    3: "construction worker",
    4: "pvc pipe",
    5: "reinforced steel",
    6: "road roller",
    7: "tower crane",
    8: "wood"
}

minet_header = {
    0: "biotite",
    1: "bornite",
    2: "chrysocolla",
    3: "malachite",
    4: "muscovite",
    5: "pyrite",
    6: "quartz"
}

datasets = ["const_res","const_bal_res","minet_res","minet_bal_res"]
headers = [const_header, const_header, minet_header, minet_header]
models = ["effib3","mobinet"]

for i in range(len(datasets)):
    dataset = datasets[i]
    header = headers[i]
    for model in models:
        with open("./results/"+dataset+"/"+model+".pkl", "rb") as file:
            data = pickle.load(file)
            mat = list(data.values())[0]['confusion_matrix']
            fig, _ = plot_confusion_matrix(mat, class_names=header.values(),figsize=(14,14))
            fig.savefig("./results/"+dataset+"/"+model+"_confusion_matrix")