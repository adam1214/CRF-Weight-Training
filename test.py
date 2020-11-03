import numpy as np
import joblib

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)

def getdata(n):
    # n為產生資料量
    x = np.arange(-5, 5.1, 10/(n-1))
    # 給定一個固定的參數，再加上隨機變動值作為雜訊，其變動值介於 +-10 之間
    y = 3*x + 2 + (np.random.rand(len(x))-0.5)*20
    return x, y

class my_MBGD:
    def __init__(self, a, b, x, y, alpha, batch_size):
        self.a = a 
        self.b = b
        self.x = x
        self.y = y 
        self.idx = 0
        self.alpha = alpha
        self.batch_size = batch_size
        
        # 使用 np.random.permutation 給定資料取出順序
        self.suffle_idx = np.random.permutation(len(x))   
        print(type(self.suffle_idx))
        self.update_batch()

    # 更新批次
    def update_batch(self):
        #每次更新時，採滾動的方式依次取出 N 筆資料
        idx = self.suffle_idx[self.idx:self.idx+self.batch_size]
        print(idx)
        print(type(idx))
        self.idx += self.batch_size

        self.x_batch = self.x[idx]
        print(self.x_batch)
        self.y_batch = self.y[idx]
        
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
    # 隨機產生一組資料
    x, y = getdata(10)
    #print(type(x))
    # 類別 MBGD 初始化
    a = -10
    b = -10
    alpha = 0.05
    batch_size = 5
    #mlclass = my_MBGD(a, b, x, y, alpha, batch_size)

    emo_dict = joblib.load('emo_all_iemocap.pkl')
    dialogs = joblib.load('dialog_iemocap.pkl')