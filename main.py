import numpy as np
import joblib
import pickle

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)

class my_MBGD:
    def __init__(self, W, X, Y, learning_rate):
        self.W = W
        self.X = X
        self.Y = Y 
        self.learning_rate = learning_rate
        
        # 使用 np.random.permutation 給定資料取出順序
        self.rand_pick_list = np.random.permutation(len(X))     
        self.rand_pick_list_index = 0
        self.update_batch()
        self.rand_pick_list_index += 1

    # 更新批次
    def update_batch(self):
        #每次更新時，採滾動的方式依次取出 N 筆資料
        print(self.rand_pick_list[self.rand_pick_list_index])
        print(self.X[self.rand_pick_list[self.rand_pick_list_index]])
        for i in range(self.rand_pick_list[self.rand_pick_list_index] - 1, 0, -1):
            if self.X[self.rand_pick_list[self.rand_pick_list_index]][:-5] != self.X[i][:-5]:
                break

        self.X_batch = self.X[i+1:self.rand_pick_list[self.rand_pick_list_index]+1]
        print(self.X_batch)
        self.Y_batch = self.Y[i+1:self.rand_pick_list[self.rand_pick_list_index]+1]
        
    # Loss function
    def mse(self):       
        sqr_err = ((self.a*self.x_batch + self.b) - self.y_batch) ** 2
        return np.mean(sqr_err)
    
    def gradient(self):        
        grad_a = 2 * np.mean((self.a*self.x_batch + self.b - self.y_batch) * (self.x_batch))
        grad_b = 2 * np.mean((self.a*self.x_batch + self.b - self.y_batch) * (1)) 
        self.update_batch()
        return grad_a, grad_b

    def update(self):
        # 計算梯度
        grad_a, grad_b = self.gradient()
        
        # 梯度更新
        self.a_old = self.a
        self.b_old = self.b
        self.a = self.a - self.alpha * grad_a
        self.b = self.b - self.alpha * grad_b
        self.loss = self.mse()

if __name__ == "__main__":
    emo_dict = joblib.load('emo_all_iemocap.pkl')
    dialogs = joblib.load('dialog_iemocap.pkl')
    X = [] #observed utterance
    Y = [] #observed emotion(only record ang, hap, neu, sad)
    W = [] #weight will be trained, 24 dims. Initialize 0 for each dim
    for i in range(24):
        W.append(0)

    for dialog in dialogs.values():
        for utt in dialog:
            if emo_dict[utt] == 'ang' or emo_dict[utt] == 'hap' or emo_dict[utt] == 'neu' or emo_dict[utt] == 'sad':
                X.append(utt)
                Y.append(emo_dict[utt])

    learning_rate = 0.05
    # 類別 MBGD 初始化
    mlclass = my_MBGD(W, X, Y, learning_rate)
    