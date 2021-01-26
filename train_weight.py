import numpy as np
import joblib
import utils
import math
import itertools
import pickle

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)

class CRF_SGD:
    def __init__(self, W, X, Y, trans_prob, learning_rate):
        self.W = W
        self.W_old = {}
        self.X = X
        self.Y = Y
        self.trans_prob = trans_prob
        self.learning_rate = learning_rate
        
        # 使用 np.random.permutation 給定資料取出順序
        self.rand_pick_list = np.random.permutation(len(X))     
        self.rand_pick_list_index = 0

        #print(self.forward_alpha(6+2, 'End'))
        #print(self.backward_beta(6, 6, 'hap'))
        #exit()
        
        self.update_batch()
    
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
        #print(Q)
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
        #print(Q)
        beta = Q[T-t-2][0]*self.trans_prob['a2End']+Q[T-t-2][1]*self.trans_prob['h2End']+Q[T-t-2][2]*self.trans_prob['n2End']+Q[T-t-2][3]*self.trans_prob['s2End']
        return beta
    
    def G_t(self, y1, y2, t): #exp{ W_y1y2 + N_py1*W_py1 + N_py2*W_py2 }
        y1 = emo_mapping_dict2[y1]
        y2 = emo_mapping_dict2[y2]

        W_y1y2 = self.W_old[y1+'2'+y2]
        if t == 1:
            utt1 = 'Start2' + y1
        else:
            utt1 = self.X_batch[t-2]
        utt2 = self.X_batch[t-1]
        
        N_py1 = out_dict[utt1][emo_index_dict[y1]]
        N_py2 = out_dict[utt2][emo_index_dict[y2]]

        W_py1 = self.W_old['p_'+y1]
        W_py2 = self.W_old['p_'+y2]

        return math.exp(W_y1y2 + N_py1*W_py1 + N_py2*W_py2)
    
    def update_batch(self): # 更新批次
        #每次更新時，採滾動的方式依次取出 N 筆資料
        #print(self.rand_pick_list[self.rand_pick_list_index])
        #print(self.X[self.rand_pick_list[self.rand_pick_list_index]])
        for i in range(self.rand_pick_list[self.rand_pick_list_index] - 1, 0, -1):
            if self.X[self.rand_pick_list[self.rand_pick_list_index]][:-5] != self.X[i][:-5]:
                break

        self.X_batch = self.X[i+1:self.rand_pick_list[self.rand_pick_list_index]+1]
        #print(self.X_batch)
        self.Y_batch = self.Y[i+1:self.rand_pick_list[self.rand_pick_list_index]+1]
        #print(self.Y_batch)
        #print(len(self.X_batch))
        #print(len(self.Y_batch))
        
        self.rand_pick_list_index += 1
        
    def gradient(self):
        emo_com = itertools.product(['ang', 'hap', 'neu', 'sad'], repeat = 2)   
        grad_W = {}
        T = len(self.X_batch)
        Z = self.forward_alpha(T+2, 'End')
        for weight_name in self.W:
            N_e1e2 = 0
            e1 = emo_mapping_dict1[weight_name[0]] #ang, hap, neu, sad, Start, pre-trained
            e2 = emo_mapping_dict1[weight_name[-1]] #ang, hap, neu, sad, End
            if e1 == 'pre-trained':
                # ex:e2為ang，將batch data中label為ang的utt在pre-trained classifier中的值相加
                for utt in self.X_batch: 
                    if e2 == emo_dict[utt]:
                        N_e1e2 = N_e1e2 + out_dict[utt][emo_index_dict[weight_name[-1]]]
                #print(e1, e2, N_e1e2)
            else:
                pre_emo = 'Start'
                current_emo = ''
                for utt in self.X_batch:
                    #print(utt, emo_dict[utt])
                    current_emo = emo_dict[utt]
                    if pre_emo == e1 and current_emo == e2:
                        N_e1e2 += 1
                    pre_emo = current_emo
                current_emo = 'End'
                if pre_emo == e1 and current_emo == e2:
                    N_e1e2 += 1
                #print(e1, e2, N_e1e2)
            
            sum_alpha_beta = 0
            for t in range(1,T,1):
                if t == 1:
                    # alpha == 0
                    sum_alpha_beta += 0    
                else:
                    for com_item in emo_com:
                        if e1 != 'pre-trained':
                            if com_item[0] == e1 and com_item[1] == e2: # N = 1
                                sum_alpha_beta = sum_alpha_beta + self.forward_alpha(t, com_item[0]) *  self.G_t(com_item[0], com_item[1], t) * self.backward_beta(t, T, com_item[1])
                            else: # N = 0
                                sum_alpha_beta += 0
                        else:
                            N = out_dict[self.X_batch[t-2]][emo_index_dict[com_item[0]]] + out_dict[self.X_batch[t-1]][emo_index_dict[com_item[1]]]
                            sum_alpha_beta = sum_alpha_beta + self.forward_alpha(t, com_item[0]) *  N * self.G_t(com_item[0], com_item[1], t) * self.backward_beta(t, T, com_item[1])
            grad_W[weight_name] = N_e1e2 - (sum_alpha_beta/Z)
        self.update_batch()
        return grad_W

    def update(self):
        # 計算梯度
        grad_W = self.gradient()
        
        for weight_name in self.W:
            self.W_old[weight_name] = self.W[weight_name]
            self.W[weight_name] = self.W[weight_name] + self.learning_rate*grad_W[weight_name]

    
if __name__ == "__main__":
    emo_mapping_dict1 = {'a':'ang', 'h':'hap', 'n':'neu', 's':'sad', 'S':'Start', 'd':'End', 'p':'pre-trained'}
    emo_mapping_dict2 = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'S', 'End':'E', 'pre-trained':'p'}
    emo_index_dict = {'a':0, 'h':1, 'n':2, 's':3, 'ang':0, 'hap':1, 'neu':2, 'sad':3}
    emo_dict = joblib.load('./data/U2U_4emo_all_iemmcap.pkl')
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    out_dict = joblib.load('./data/outputs.pkl')

    trans_prob = utils.emo_trans_prob_BI_without_softmax(emo_dict, dialogs)
    
    # pre-trained calssifier中增加4項，以logits計算
    out_dict['Start2a'] = math.log(trans_prob['Start2a']/(1-trans_prob['Start2a']), 2)
    out_dict['Start2h'] = math.log(trans_prob['Start2h']/(1-trans_prob['Start2h']), 2)
    out_dict['Start2n'] = math.log(trans_prob['Start2n']/(1-trans_prob['Start2n']), 2)
    out_dict['Start2s'] = math.log(trans_prob['Start2s']/(1-trans_prob['Start2s']), 2)
    
    # emo_dict = joblib.load('./data/emo_all_iemocap.pkl')

    
    # print(out_dict)
    X = [] #observed utterance
    Y = [] #observed emotion(only record ang, hap, neu, sad)

    # Weight:relation between emos and
    # Weight:relation between pre-trained & emos
    W = { 'Start2a':0, 'Start2h':0, 'Start2n':0, 'Start2s':0, \
          'a2a':0, 'a2h':0, 'a2n':0, 'a2s':0, \
          'h2a':0, 'h2h':0, 'h2n':0, 'h2s':0, \
          'n2a':0, 'n2h':0, 'n2n':0, 'n2s':0, \
          's2a':0, 's2h':0, 's2n':0, 's2s':0, \
          'a2End':0, 'h2End':0, 'n2End':0, 's2End':0, \
          'p_a':0, 'p_h':0, 'p_n':0, 'p_s':0 
        }
    
    for dialog in dialogs.values():
        for utt in dialog:
            if emo_dict[utt] == 'ang' or emo_dict[utt] == 'hap' or emo_dict[utt] == 'neu' or emo_dict[utt] == 'sad':
                X.append(utt)
                Y.append(emo_dict[utt])
    
    learning_rate = 0.0001
    CRF_model = CRF_SGD(W, X, Y, trans_prob,learning_rate) # 類別 CRF_SGD 初始化
    
    for i in range(1, 5001, 1):
        print(i)
        CRF_model.update()
        #print(CRF_model.W_old['h2h'],CRF_model.W['h2h'])
    # print(CRF_model.W)
    file=open('weight.pickle','wb')
    pickle.dump(CRF_model.W, file)
    file.close()
    