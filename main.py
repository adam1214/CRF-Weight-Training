import numpy as np
import joblib
import utils

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)

class CRF_SGD:
    def __init__(self, W, X, Y, trans_prob, learning_rate):
        self.W = W
        self.X = X
        self.Y = Y
        self.trans_prob = trans_prob
        self.learning_rate = learning_rate
        
        # 使用 np.random.permutation 給定資料取出順序
        self.rand_pick_list = np.random.permutation(len(X))     
        self.rand_pick_list_index = 0

        #print(self.forward_alpha(6+2, 'End'))
        #print(self.backward_beta(6, 6, 'hap'))
        exit()
        
        self.update_batch()
        self.rand_pick_list_index += 1
    
    def forward_alpha(self, t, y1):
        #alpha為第0秒到第(t-1)秒的第y1個情緒之所有可能路徑機率和
        t -= 1
        if t == 0:
            return 0
        else:
            Q = [([0]*4) for i in range(t)] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        # 第一個時間點的情緒(prior prob.)
        Q[0][0] = self.trans_prob['Start2a']
        Q[0][1] = self.trans_prob['Start2h']
        Q[0][2] = self.trans_prob['Start2n']
        Q[0][3] = self.trans_prob['Start2s']

        for i in range(1, t - 1, 1):
            for j in range(0, 4, 1):
                if j == 0:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2a']+Q[i-1][1]*self.trans_prob['h2a']+Q[i-1][2]*self.trans_prob['n2a']+Q[i-1][3]*self.trans_prob['s2a']
                elif j == 1:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2h']+Q[i-1][1]*self.trans_prob['h2h']+Q[i-1][2]*self.trans_prob['n2h']+Q[i-1][3]*self.trans_prob['s2h']
                elif j == 2:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2n']+Q[i-1][1]*self.trans_prob['h2n']+Q[i-1][2]*self.trans_prob['n2n']+Q[i-1][3]*self.trans_prob['s2n']
                elif j == 3:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2s']+Q[i-1][1]*self.trans_prob['h2s']+Q[i-1][2]*self.trans_prob['n2s']+Q[i-1][3]*self.trans_prob['s2s']
        
        if y1 == 'ang':
            alpha = Q[t-2][0]*self.trans_prob['a2a']+Q[t-2][1]*self.trans_prob['h2a']+Q[t-2][2]*self.trans_prob['n2a']+Q[t-2][3]*self.trans_prob['s2a']
        elif y1 == 'hap':
            alpha = Q[t-2][0]*self.trans_prob['a2h']+Q[t-2][1]*self.trans_prob['h2h']+Q[t-2][2]*self.trans_prob['n2h']+Q[t-2][3]*self.trans_prob['s2h']
        elif y1 == 'neu':
            alpha = Q[t-2][0]*self.trans_prob['a2n']+Q[t-2][1]*self.trans_prob['h2n']+Q[t-2][2]*self.trans_prob['n2n']+Q[t-2][3]*self.trans_prob['s2n']
        elif y1 == 'sad':
            alpha = Q[t-2][0]*self.trans_prob['a2s']+Q[t-2][1]*self.trans_prob['h2s']+Q[t-2][2]*self.trans_prob['n2s']+Q[t-2][3]*self.trans_prob['s2s']
        elif y1 == 'End': # estimate Z(T)
            alpha = Q[t-2][0]*self.trans_prob['a2End']+Q[t-2][1]*self.trans_prob['h2End']+Q[t-2][2]*self.trans_prob['n2End']+Q[t-2][3]*self.trans_prob['s2End']
        print(Q)
        return alpha

    def backward_beta(self, t, T, y2):
        #beta為第t秒的第y2個情緒到第(T+1)秒之所有可能路徑機率和
        T += 1
        Q = [([0]*4) for i in range(T-t)] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # 第一個時間點的情緒(prior prob.)
        if y2 == 'ang':    
            key = 'a'
        elif y2 == 'hap':
            key = 'h'
        elif y2 == 'neu':
            key = 'n'
        elif y2 == 'sad':
            key = 's'
        Q[0][0] = self.trans_prob[key+'2a']
        Q[0][1] = self.trans_prob[key+'2h']
        Q[0][2] = self.trans_prob[key+'2n']
        Q[0][3] = self.trans_prob[key+'2s']

        for i in range(1, T - t - 1, 1):
            for j in range(0, 4, 1):
                if j == 0:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2a']+Q[i-1][1]*self.trans_prob['h2a']+Q[i-1][2]*self.trans_prob['n2a']+Q[i-1][3]*self.trans_prob['s2a']
                elif j == 1:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2h']+Q[i-1][1]*self.trans_prob['h2h']+Q[i-1][2]*self.trans_prob['n2h']+Q[i-1][3]*self.trans_prob['s2h']
                elif j == 2:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2n']+Q[i-1][1]*self.trans_prob['h2n']+Q[i-1][2]*self.trans_prob['n2n']+Q[i-1][3]*self.trans_prob['s2n']
                elif j == 3:
                    Q[i][j] = Q[i-1][0]*self.trans_prob['a2s']+Q[i-1][1]*self.trans_prob['h2s']+Q[i-1][2]*self.trans_prob['n2s']+Q[i-1][3]*self.trans_prob['s2s']
        print(Q)
        beta = Q[T-t-2][0]*self.trans_prob['a2End']+Q[T-t-2][1]*self.trans_prob['h2End']+Q[T-t-2][2]*self.trans_prob['n2End']+Q[T-t-2][3]*self.trans_prob['s2End']
        return beta
    
    def update_batch(self): # 更新批次
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

def main():
    emo_dict = joblib.load('./data/U2U_4emo_all_iemmcap.pkl')
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    out_dict = joblib.load('./data/outputs.pkl')

    trans_prob = utils.emo_trans_prob_BI_without_softmax(emo_dict, dialogs)

    emo_dict = joblib.load('./data/emo_all_iemocap.pkl')

    
    # print(out_dict)
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
    # 類別 CRF_SGD 初始化
    CRF_model = CRF_SGD(W, X, Y, trans_prob,learning_rate)
    
    
if __name__ == "__main__":
    main()
