import os
import math
import random
import shutil

train_ratio = 0.8
test_ratio = 1-train_ratio

path = './data'
test_path = "./test"

datasets = {"const data":"const data test","Minet 5640 Images":"Minet 5640 test"}
for dataset in datasets:
    test_dataset = datasets[dataset]
    os.mkdir(test_path+"/"+test_dataset)
    classes = os.listdir(path+"/"+dataset)
    if ".DS_Store" in  classes: classes.remove(".DS_Store")

    for c in classes:
        ls = os.listdir(path+"/"+dataset+"/"+c)
        l = math.ceil(test_ratio * len(ls))
        print(path+"/"+dataset+"/"+c," :- ",len(ls)," ",l)
        os.mkdir(test_path+"/"+test_dataset+"/"+c)

        for i in range(l):
            f = random.choice(ls)
            shutil.copyfile(
                path+"/"+dataset+"/"+c+"/"+f, 
                test_path+"/"+test_dataset+"/"+c+"/"+f)
            ls.remove(f)
