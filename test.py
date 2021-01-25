import pickle

if __name__ == "__main__":
    with open('weight.pickle','rb') as file:
        W=pickle.load(file)
    print(W)