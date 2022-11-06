import pickle as pkl
import seaborn
from pandas import DataFrame as df
import os
from matplotlib import pyplot as plt

seaborn.set(rc={'figure.figsize':(16,16)})

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


def get_pred_vals(conf_mat):
    correct = []
    wrong = []

    for index in range(len(conf_mat)):
        correct.append(conf_mat[index][index])
    
    for row in range(len(conf_mat)):
        sum = 0
        for col in range(len(conf_mat)):
            if row==col:
                continue
            sum += conf_mat[row][col]
        wrong.append(sum)
    
    data = []

    for index in range(len(correct)):
        data.append([index, correct[index], 'correct'])
        data.append([index, wrong[index], 'wrong'])

    dataframe = df(data, columns=['class', 'number of images', 'prediction'])
    
    dataframe['class'] = dataframe['class'].map(lambda x: const_header[x])

    return dataframe


def output_graph(path, outpath, name, header):
    with open(path, 'rb') as file:
        summary = pkl.load(file)

    vals = get_pred_vals(list(summary.values())[0]['confusion_matrix'])
    
    graph = seaborn.barplot(data=vals, x = 'class', y='number of images', hue = 'prediction', errorbar=None)
    if header == 'minet':
        graph.set_xticklabels(list(minet_header.values()), rotation = 90)
    else: 
        graph.set_xticklabels(list(const_header.values()), rotation = 90)

    graph.get_figure().savefig(outpath+'/'+name)

    plt.clf()

if __name__ == '__main__':
    base_path = 'results'

    for dir in os.listdir(base_path):
        for file in os.listdir(os.path.join(base_path, dir)):
            if file.split('.')[-1]=='png':
                continue
            if dir.split('_')[0] == 'minet':
                output_graph(os.path.join(base_path, dir, file), './graphs', dir+'_'+file.split('.')[0], 'minet')
            else:
                output_graph(os.path.join(base_path, dir, file), './graphs', dir+'_'+file.split('.')[0], 'const')
