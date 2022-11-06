import pickle

if __name__=='__main__':
    
    with open('./results/const_bal', 'rb') as file:
        summary =  pickle.load(file)
    
    print(summary)

