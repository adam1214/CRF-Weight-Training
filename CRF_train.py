import numpy as np
import joblib
import math
import itertools
import pickle
import seaborn as sn
import matplotlib.pyplot as plt
import argparse

import utils
import CRF_test

from argparse import RawTextHelpFormatter

# 給定隨機種子，使每次執行結果保持一致
np.random.seed(1)

class CRF_SGD:
    def __init__(self, W, X, Y, trans_prob_no_spk_info, trans_prob_inter, trans_prob_intra, out_dict, learning_rate):
        self.W = W
        self.W_np = np.zeros((28))
        self.W_old = {}
        for weight_name in self.W:
            self.W_old[weight_name] = self.W[weight_name]
        self.X = X
        self.Y = Y
        self.trans_prob_no_spk_info = trans_prob_no_spk_info
        self.trans_prob_inter = trans_prob_inter
        self.trans_prob_intra = trans_prob_intra
        self.trans_prob = {}
        self.out_dict = out_dict
        self.learning_rate = learning_rate
        
        # 使用 np.random.permutation 給定資料取出順序
        self.rand_pick_list = np.random.permutation(len(X))     
        self.rand_pick_list_index = 0
        
        if args.speaker_info_train == 1:
            self.X_M_batch = []
            self.X_F_batch = []
            self.Y_M_batch = []
            self.Y_F_batch = []
            self.batch_list_index = 0
            self.trans_prob_inter = {}

        self.update_batch()
    
    def forward_alpha(self, t, y1):
        #alpha為第0秒到第(t-1)秒的第y1個情緒之所有可能路徑機率和
        t -= 1
        if t == 0:
            return math.exp(0)
        else:
            Q = [([0]*4) for i in range(t)] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        utt = self.X_batch[0]
        pre_spk = utt[-4]
        if args.speaker_info_train == 0:
            self.trans_prob = self.trans_prob_no_spk_info
        else:
            self.trans_prob = self.trans_prob_intra
        # 第一個時間點 (transition prob. * weight + emission prob. * weight) => (f_j*w_j)
        Q[0][0] = math.exp(self.trans_prob['Start2a']*self.W_old['Start2a'] + self.out_dict[utt][0]*self.W_old['p_a'])
        Q[0][1] = math.exp(self.trans_prob['Start2h']*self.W_old['Start2h'] + self.out_dict[utt][1]*self.W_old['p_h'])
        Q[0][2] = math.exp(self.trans_prob['Start2n']*self.W_old['Start2n'] + self.out_dict[utt][2]*self.W_old['p_n'])
        Q[0][3] = math.exp(self.trans_prob['Start2s']*self.W_old['Start2s'] + self.out_dict[utt][3]*self.W_old['p_s'])

        for i in range(1, t - 1, 1):
            utt = self.X_batch[i]
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            elif utt[-4] == pre_spk:
                self.trans_prob = self.trans_prob_intra
            elif utt[-4] != pre_spk:
                self.trans_prob = self.trans_prob_inter
            for j in range(0, 4, 1):
                if j == 0:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2a']*self.W_old['a2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2a']*self.W_old['h2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2a']*self.W_old['n2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2a']*self.W_old['s2a'] + self.out_dict[utt][0]*self.W_old['p_a'])
                elif j == 1:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2h']*self.W_old['a2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2h']*self.W_old['h2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2h']*self.W_old['n2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2h']*self.W_old['s2h'] + self.out_dict[utt][1]*self.W_old['p_h'])
                elif j == 2:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2n']*self.W_old['a2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2n']*self.W_old['h2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2n']*self.W_old['n2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2n']*self.W_old['s2n'] + self.out_dict[utt][2]*self.W_old['p_n'])
                elif j == 3:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2s']*self.W_old['a2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2s']*self.W_old['h2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2s']*self.W_old['n2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2s']*self.W_old['s2s'] + self.out_dict[utt][3]*self.W_old['p_s'])
            pre_spk = utt[-4]

        if y1 == 'ang':
            utt = self.X_batch[t-1]
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            elif utt[-4] == pre_spk:
                self.trans_prob = self.trans_prob_intra
            elif utt[-4] != pre_spk:
                self.trans_prob = self.trans_prob_inter
            alpha = Q[t-2][0]*math.exp(self.trans_prob['a2a']*self.W_old['a2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                    Q[t-2][1]*math.exp(self.trans_prob['h2a']*self.W_old['h2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                    Q[t-2][2]*math.exp(self.trans_prob['n2a']*self.W_old['n2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                    Q[t-2][3]*math.exp(self.trans_prob['s2a']*self.W_old['s2a'] + self.out_dict[utt][0]*self.W_old['p_a'])
        elif y1 == 'hap':
            utt = self.X_batch[t-1]
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            elif utt[-4] == pre_spk:
                self.trans_prob = self.trans_prob_intra
            elif utt[-4] != pre_spk:
                self.trans_prob = self.trans_prob_inter
            alpha = Q[t-2][0]*math.exp(self.trans_prob['a2h']*self.W_old['a2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                    Q[t-2][1]*math.exp(self.trans_prob['h2h']*self.W_old['h2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                    Q[t-2][2]*math.exp(self.trans_prob['n2h']*self.W_old['n2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                    Q[t-2][3]*math.exp(self.trans_prob['s2h']*self.W_old['s2h'] + self.out_dict[utt][1]*self.W_old['p_h'])
        elif y1 == 'neu':
            utt = self.X_batch[t-1]
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            elif utt[-4] == pre_spk:
                self.trans_prob = self.trans_prob_intra
            elif utt[-4] != pre_spk:
                self.trans_prob = self.trans_prob_inter
            alpha = Q[t-2][0]*math.exp(self.trans_prob['a2n']*self.W_old['a2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                    Q[t-2][1]*math.exp(self.trans_prob['h2n']*self.W_old['h2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                    Q[t-2][2]*math.exp(self.trans_prob['n2n']*self.W_old['n2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                    Q[t-2][3]*math.exp(self.trans_prob['s2n']*self.W_old['s2n'] + self.out_dict[utt][2]*self.W_old['p_n'])
        elif y1 == 'sad':
            utt = self.X_batch[t-1]
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            elif utt[-4] == pre_spk:
                self.trans_prob = self.trans_prob_intra
            elif utt[-4] != pre_spk:
                self.trans_prob = self.trans_prob_inter
            alpha = Q[t-2][0]*math.exp(self.trans_prob['a2s']*self.W_old['a2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                    Q[t-2][1]*math.exp(self.trans_prob['h2s']*self.W_old['h2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                    Q[t-2][2]*math.exp(self.trans_prob['n2s']*self.W_old['n2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                    Q[t-2][3]*math.exp(self.trans_prob['s2s']*self.W_old['s2s'] + self.out_dict[utt][3]*self.W_old['p_s'])
        elif y1 == 'End': # estimate Z(T)
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            else:
                self.trans_prob = self.trans_prob_intra
            alpha = Q[t-2][0]*math.exp(self.trans_prob['a2End']*self.W_old['a2End']) + \
                    Q[t-2][1]*math.exp(self.trans_prob['h2End']*self.W_old['h2End']) + \
                    Q[t-2][2]*math.exp(self.trans_prob['n2End']*self.W_old['n2End']) + \
                    Q[t-2][3]*math.exp(self.trans_prob['s2End']*self.W_old['s2End'])
        #print(Q)
        return alpha

    def create_alpha_lookup_dict(self, T):
        alpha_lookup_dict = {}
        for t in range(1, T+1, 1):
            if t == 1:
                alpha_lookup_dict[t] = {'Start':self.forward_alpha(t, 'Start')}
            else:
                alpha_lookup_dict[t] = {'ang':self.forward_alpha(t, 'ang'), \
                                        'hap':self.forward_alpha(t, 'hap'), \
                                        'neu':self.forward_alpha(t, 'neu'), \
                                        'sad':self.forward_alpha(t, 'sad')  }
        return alpha_lookup_dict
    
    def backward_beta(self, t, T, y2):
        #beta為第t秒的第y2個情緒到第(T+1)秒之所有可能路徑機率和
        T += 1
        Q = [([0]*4) for i in range(T-t)] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        
        utt = self.X_batch[t-1]
        utt_pre = self.X_batch[t-2]
        if args.speaker_info_train == 0:
            self.trans_prob = self.trans_prob_no_spk_info
        elif utt[-4] == utt_pre[-4]:
            self.trans_prob = self.trans_prob_intra
        elif utt[-4] != utt_pre[-4]:
            self.trans_prob = self.trans_prob_inter      
        # 第一個時間點 (transition prob. * weight + emission prob. * weight) => (f_j*w_j)
        if y2 == 'ang':
            key = 'a'
        elif y2 == 'hap':
            key = 'h'
        elif y2 == 'neu':
            key = 'n'
        elif y2 == 'sad':
            key = 's'
        Q[0][0] = math.exp(self.trans_prob[key+'2a']*self.W_old[key+'2a'] + self.out_dict[utt][0]*self.W_old['p_a'])
        Q[0][1] = math.exp(self.trans_prob[key+'2h']*self.W_old[key+'2h'] + self.out_dict[utt][1]*self.W_old['p_h'])
        Q[0][2] = math.exp(self.trans_prob[key+'2n']*self.W_old[key+'2n'] + self.out_dict[utt][2]*self.W_old['p_n'])
        Q[0][3] = math.exp(self.trans_prob[key+'2s']*self.W_old[key+'2s'] + self.out_dict[utt][3]*self.W_old['p_s'])
        
        pre_spk = utt[-4]
        for i in range(1, T - t - 1, 1):
            utt = self.X_batch[t-1+i]
            if args.speaker_info_train == 0:
                self.trans_prob = self.trans_prob_no_spk_info
            elif utt[-4] == pre_spk:
                self.trans_prob = self.trans_prob_intra
            elif utt[-4] != pre_spk:
                self.trans_prob = self.trans_prob_inter
            for j in range(0, 4, 1):
                if j == 0:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2a']*self.W_old['a2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2a']*self.W_old['h2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2a']*self.W_old['n2a'] + self.out_dict[utt][0]*self.W_old['p_a']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2a']*self.W_old['s2a'] + self.out_dict[utt][0]*self.W_old['p_a'])
                elif j == 1:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2h']*self.W_old['a2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2h']*self.W_old['h2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2h']*self.W_old['n2h'] + self.out_dict[utt][1]*self.W_old['p_h']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2h']*self.W_old['s2h'] + self.out_dict[utt][1]*self.W_old['p_h'])
                elif j == 2:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2n']*self.W_old['a2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2n']*self.W_old['h2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2n']*self.W_old['n2n'] + self.out_dict[utt][2]*self.W_old['p_n']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2n']*self.W_old['s2n'] + self.out_dict[utt][2]*self.W_old['p_n'])
                elif j == 3:
                    Q[i][j] = Q[i-1][0]*math.exp(self.trans_prob['a2s']*self.W_old['a2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                              Q[i-1][1]*math.exp(self.trans_prob['h2s']*self.W_old['h2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                              Q[i-1][2]*math.exp(self.trans_prob['n2s']*self.W_old['n2s'] + self.out_dict[utt][3]*self.W_old['p_s']) + \
                              Q[i-1][3]*math.exp(self.trans_prob['s2s']*self.W_old['s2s'] + self.out_dict[utt][3]*self.W_old['p_s'])
            pre_spk = utt[-4]
        #print(Q)
        if args.speaker_info_train == 0:
            self.trans_prob = self.trans_prob_no_spk_info
        else:
            self.trans_prob = self.trans_prob_intra
        beta = Q[T-t-2][0]*math.exp(self.trans_prob['a2End']*self.W_old['a2End']) + \
               Q[T-t-2][1]*math.exp(self.trans_prob['h2End']*self.W_old['h2End']) + \
               Q[T-t-2][2]*math.exp(self.trans_prob['n2End']*self.W_old['n2End']) + \
               Q[T-t-2][3]*math.exp(self.trans_prob['s2End']*self.W_old['s2End'])
        return beta
    
    def create_beta_lookup_dict(self, T):
        beta_lookup_dict = {}
        for t in range(1, T+1, 1):
            beta_lookup_dict[t] = {'ang':self.backward_beta(t, T, 'ang'), \
                                   'hap':self.backward_beta(t, T, 'hap'), \
                                   'neu':self.backward_beta(t, T, 'neu'), \
                                   'sad':self.backward_beta(t, T, 'sad')  }
        return beta_lookup_dict

    def G_t(self, y1, y2, t): #exp{ N_y1y2*W_y1y2 + N_py1*W_py1 + N_py2*W_py2 }
        y1 = emo_mapping_dict2[y1] #Start, a, h, n, s
        y2 = emo_mapping_dict2[y2] #a, h, n, s

        if t == 1:
            utt1 = 'Start'
            N_py1 = 0
            W_py1 = 0
        else:
            utt1 = self.X_batch[t-2]
            N_py1 = self.out_dict[utt1][emo_index_dict[y1]]
            W_py1 = self.W_old['p_'+y1]
        
        utt2 = self.X_batch[t-1]

        if args.speaker_info_train == 0:
            self.trans_prob = self.trans_prob_no_spk_info
        elif utt1[-4] == utt2[-4] or utt1[-4] == 't':
            self.trans_prob = self.trans_prob_intra
        else:
            self.trans_prob = self.trans_prob_inter
        N_y1y2 = self.trans_prob[y1+'2'+y2]
        W_y1y2 = self.W_old[y1+'2'+y2]

        N_py2 = self.out_dict[utt2][emo_index_dict[y2]]
        W_py2 = self.W_old['p_'+y2]
        
        return math.exp(N_y1y2*W_y1y2 + N_py1*W_py1 + N_py2*W_py2)

    def nested_dict(self, dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    def create_G_t_lookup_dict(self, T, emo_com_list):
        G_t_lookup_dict = {}
        for t in range(1, T+1, 1):
            if t == 1:
                self.nested_dict(G_t_lookup_dict, [t, ('Start', 'ang')], self.G_t('Start', 'ang', t))
                self.nested_dict(G_t_lookup_dict, [t, ('Start', 'hap')], self.G_t('Start', 'hap', t))
                self.nested_dict(G_t_lookup_dict, [t, ('Start', 'neu')], self.G_t('Start', 'neu', t))
                self.nested_dict(G_t_lookup_dict, [t, ('Start', 'sad')], self.G_t('Start', 'sad', t))
            else:
                for e_com in emo_com_list:
                    self.nested_dict(G_t_lookup_dict, [t, e_com], self.G_t(e_com[0], e_com[1], t))
        return G_t_lookup_dict
    
    def split_dialog(self):
        for i in range(0,len(self.X_batch),1):
            utt = self.X_batch[i]
            emo = self.Y_batch[i]
            if utt[-4] == 'F':
                self.X_F_batch.append(utt)
                self.Y_F_batch.append(emo)
            else:
                self.X_M_batch.append(utt)
                self.Y_M_batch.append(emo)

    def update_batch(self): # 更新批次
        #每次更新時，採滾動的方式依次取出 N 筆資料
        #print(self.rand_pick_list[self.rand_pick_list_index])
        #print(self.X[self.rand_pick_list[self.rand_pick_list_index]])
        for utt_index in range(self.rand_pick_list[self.rand_pick_list_index] - 1, -2, -1):
            if self.X[self.rand_pick_list[self.rand_pick_list_index]][:-5] != self.X[utt_index][:-5]:
                break

        self.X_batch = self.X[utt_index+1:self.rand_pick_list[self.rand_pick_list_index]+1]
        #print(self.X_batch)
        self.Y_batch = self.Y[utt_index+1:self.rand_pick_list[self.rand_pick_list_index]+1]
        #print(self.Y_batch)
        if args.speaker_info_train == 1:
            self.X_M_batch.clear()
            self.X_F_batch.clear()
            self.Y_M_batch.clear()
            self.Y_F_batch.clear()
            self.split_dialog()
            self.X_batch_list = [self.X_M_batch, self.X_F_batch]
            self.Y_batch_list = [self.Y_M_batch, self.Y_F_batch]
            
            self.batch_list_index = 0
            self.X_batch = self.X_batch_list[self.batch_list_index]
            self.Y_batch = self.Y_batch_list[self.batch_list_index]
            if len(self.X_batch) == 0:
                self.batch_list_index += 1
                self.X_batch = self.X_batch_list[self.batch_list_index]
                self.Y_batch = self.Y_batch_list[self.batch_list_index]

        #print(self.X_F_batch)
        #print(self.Y_F_batch)
        #print(len(self.X_batch))
        #print(len(self.Y_batch))
        
        self.rand_pick_list_index += 1
        
    def gradient(self):
        emo_com = itertools.product(['ang', 'hap', 'neu', 'sad'], repeat = 2)   
        emo_com_list = [item for item in emo_com] 
        Start_emo_com_list = [('Start', 'ang'), ('Start', 'hap'), ('Start', 'neu'), ('Start', 'sad')]
        T = len(self.X_batch)
        Z = self.forward_alpha(T+2, 'End')

        N_e1e2 = np.zeros((28))
        N_internal = np.zeros((T,16,28)) # [combination][feature][time]
        j = 0
        for weight_name in self.W:
            e1 = emo_mapping_dict1[weight_name[0]] #ang, hap, neu, sad, Start, pre-trained
            e2 = emo_mapping_dict1[weight_name[-1]] #ang, hap, neu, sad, End
            if e1 == 'pre-trained': # part2 weight feature(N) extraction
                # ex:e2為ang，將batch data中label為ang的utt在pre-trained classifier中的值相加
                for utt in self.X_batch: 
                    if e2 == emo_dict[utt]:
                        N_e1e2[j] = N_e1e2[j] + self.out_dict[utt][emo_index_dict[weight_name[-1]]]
            else: # part1 weight feature(N) extraction
                pre_emo = 'Start'
                current_emo = ''

                pre_spk = ''
                cur_spk = ''
                for utt in self.X_batch:
                    #print(utt, emo_dict[utt])
                    current_emo = emo_dict[utt]
                    cur_spk = utt[-4]
                    if args.speaker_info_train == 0:
                        self.trans_prob = self.trans_prob_no_spk_info
                    elif cur_spk == pre_spk or pre_spk == '':
                        self.trans_prob = self.trans_prob_intra
                    else:
                        self.trans_prob = self.trans_prob_inter

                    if pre_emo == e1 and current_emo == e2:
                        N_e1e2[j] += self.trans_prob[emo_mapping_dict2[e1]+'2'+emo_mapping_dict2[e2]]

                    pre_emo = current_emo
                    pre_spk = cur_spk

                current_emo = 'End'
                cur_spk = 'End'
                if args.speaker_info_train == 0:
                    self.trans_prob = self.trans_prob_no_spk_info
                else:
                    self.trans_prob = self.trans_prob_intra
                if pre_emo == e1 and current_emo == e2:
                    N_e1e2[j] += self.trans_prob[emo_mapping_dict2[e1]+'2'+emo_mapping_dict2[e2]]
            
            for t in range(1,T+1,1):
                c = 0
                if t == 1:
                    tmp_emo_com_list = Start_emo_com_list #len is 4 (c = 0~3)
                else:
                    tmp_emo_com_list = emo_com_list #len is 16 (c = 0~15)

                for emo_com_item in tmp_emo_com_list:
                    if e1 != 'pre-trained': #part 2:relation between emos (internal feature extraction)
                        if emo_com_item[0] == e1 and emo_com_item[1] == e2:
                            if args.speaker_info_train == 0:
                                self.trans_prob = self.trans_prob_no_spk_info
                            elif t == 1 or (self.X_batch[t-2])[-4] == (self.X_batch[t-1])[-4]:
                                self.trans_prob = self.trans_prob_intra
                            else:
                                self.trans_prob = self.trans_prob_inter
                            N_internal[t-1][c][j] = self.trans_prob[emo_mapping_dict2[e1]+'2'+emo_mapping_dict2[e2]] # transition prob.
                        else:
                            N_internal[t-1][c][j] = 0 # transition prob. == 0
                    else: #part 1:relation between pre-trained & emos (internal feature extraction)
                        if emo_com_item[1] == e2:
                            N_internal[t-1][c][j] = self.out_dict[self.X_batch[t-1]][emo_index_dict[emo_com_item[1]]] # emission prob.
                        else:
                            N_internal[t-1][c][j] = 0
                    c += 1
            j += 1
        
        alpha_lookup_dict = self.create_alpha_lookup_dict(T)
        beta_lookup_dict = self.create_beta_lookup_dict(T)
        G_t_lookup_dict = self.create_G_t_lookup_dict(T, emo_com_list)
        sum_alpha_beta_np = np.zeros((28))

        for t in range(1,T+1,1):
            c = 0
            if t == 1:
                tmp_emo_com_list = Start_emo_com_list
            else:
                tmp_emo_com_list = emo_com_list

            for emo_com_item in tmp_emo_com_list:
                sum_alpha_beta_np = sum_alpha_beta_np + alpha_lookup_dict[t][emo_com_item[0]] * N_internal[t-1][c] * G_t_lookup_dict[t][emo_com_item] * beta_lookup_dict[t][emo_com_item[1]]
                c += 1

        sum_alpha_beta_np = sum_alpha_beta_np / Z
        grad_W_np = N_e1e2 - sum_alpha_beta_np
        if args.speaker_info_train == 1 and self.batch_list_index == 0:
            self.batch_list_index += 1
            self.X_batch = self.X_batch_list[self.batch_list_index]
            if len(self.X_batch) == 0:
                self.update_batch()
        else:
            self.update_batch()
        return grad_W_np

    def update(self):
        # 計算梯度
        grad_W_np = self.gradient()
        self.W_np = np.array(list(self.W.values()))

        self.W_np = self.W_np + self.learning_rate * grad_W_np

        j = 0
        for weight_name in self.W:
            self.W_old[weight_name] = self.W[weight_name]
            self.W[weight_name] = self.W_np[j]
            j += 1

        if args.speaker_info_train == 1 and self.batch_list_index == 1:
            self.update()
        
def test_acc(S1_Weight, S2_Weight, S3_Weight, S4_Weight, S5_Weight):
    predict = []
    for _, dia in enumerate(dialogs):
        Session_num = dialogs[dia][0][0:5]
        if Session_num == 'Ses01':
            W = S1_Weight
        elif Session_num == 'Ses02':
            W = S2_Weight
        elif Session_num == 'Ses03':
            W = S3_Weight
        elif Session_num == 'Ses04':
            W = S4_Weight
        elif Session_num == 'Ses05':
            W = S5_Weight
        
        if args.concatenation == 1:
            concat_dialog = dialogs[dia] + dialogs[dia]
        else:
            concat_dialog = dialogs[dia]
        
        if args.inter_intra_test == 'inter':
            predict += CRF_test.viterbi_inter(W, concat_dialog, no_speaker_info_emo_trans_prob_dict[Session_num], inter_emo_trans_prob_dict[Session_num], intra_emo_trans_prob_dict[Session_num], out_dict, args.concatenation, args.speaker_info_train)
        elif args.inter_intra_test == 'intra':
            predict += CRF_test.viterbi_intra(W, concat_dialog, no_speaker_info_emo_trans_prob_dict[Session_num], intra_emo_trans_prob_dict[Session_num], out_dict, args.concatenation, args.speaker_info_train)
    
    uar, acc, conf = utils.evaluate(predict, label)
    print('DED performance: uar: %.3f, acc: %.3f' % (uar, acc))
    print(conf)
    return uar, acc, conf

def plot_dynamic_line_chart(uars, accs, Iter, iteration, uar ,acc):
    global uars_arr, accs_arr, iters, ann_list
    
    iters[0] = iters[1]
    iters[1] = Iter

    uars[0] = uars[1]
    uars[1] = uar

    accs[0] = accs[1]
    accs[1] = acc

    plt.plot(iters, uars, c='red', marker='s', label='UAR', lw=1, ms=3)
    plt.plot(iters, accs, c='blue', marker='s', label='ACC', lw=1, ms=3)
    
    #print(ann_list)
    for _, ann in enumerate(ann_list):
        ann.remove()
    ann_list[:] = []

    uars_arr = np.append(uars_arr, uar)
    accs_arr = np.append(accs_arr, acc)

    max_uars_index = np.argmax(uars_arr) #max value uars_arr index
    max_accs_index = np.argmax(accs_arr) #max value accs_arr index
    
    show_uar_max = '('+str(max_uars_index + 1) + ', ' + str(uars_arr[max_uars_index])+')'
    show_acc_max = '('+str(max_accs_index + 1) + ', ' + str(accs_arr[max_accs_index])+')'

    ann = plt.annotate(show_uar_max, xytext=(max_uars_index + 1, uars_arr[max_uars_index]), xy = (max_uars_index + 1, uars_arr[max_uars_index]), c = 'red')
    ann_list.append(ann)
    ann = plt.annotate(show_acc_max, xytext=(max_accs_index + 1, accs_arr[max_accs_index] + 0.02), xy = (max_accs_index + 1, accs_arr[max_accs_index]), c = 'blue')
    ann_list.append(ann)

    print('iteration ' + str(max_uars_index + 1) + ' with the best UAR:' + str(uars_arr[max_uars_index]))
    print('iteration ' + str(max_accs_index + 1) + ' with the best ACC:' + str(accs_arr[max_accs_index]))
    
    if Iter == 1:
        plt.legend(loc = 'upper left')
        plt.xlabel('Training Iteration')
        plt.ylabel('Probability')
        
        plt.title(diagram_title, fontsize=10)
    plt.pause(0.0001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("-i", "--iteration", type=int, help="Set parameter update times.", default = 3000)
    parser.add_argument("-l", "--learning_rate", type=float, help="Set learning rate.", default = 0.000001)
    parser.add_argument("-d", "--dataset", type=str, help="Set the dataset to be used for training:\n\tOption 1:Original\n\tOption 2:C2C (Class to class mapping by pre-trained classifier)\n\tOption 3:U2U (Utt to Utt mapping by pre-trained classifier)", default = "Original")
    parser.add_argument("-c", "--concatenation", type=int, help="When predicting a dialog, do you want to duplicate it 2 times and concatenate them together? 1 is yes, 0 is no", default = 0) # no concatenation is better
    parser.add_argument("-n", "--inter_intra_test", type=str, help="When predicting a dialog, use intraspeaker emotion flow or interspeaker emotion change.", default = "intra")
    parser.add_argument("-s", "--speaker_info_train", type=int, help="When estimating emotion transition probabilities, do you want to consider speakers information?\n\t0:not consider speaker info\n\t1:consider intra-speaker only\n\t2:consider intra-speaker & inter-speaker", default = 1)

    args = parser.parse_args()
    diagram_title = 'Learning Rate:' + str(args.learning_rate) + '#####' + str(args.iteration) + ' Iteration#####' + args.dataset + ' dataset\n'
    if args.concatenation == 1:
        diagram_title += 'With concatenation#####'
    else:
        diagram_title += 'Without concatenation#####'
    diagram_title += 'Viterbi algo. with '
    diagram_title += args.inter_intra_test
    if args.speaker_info_train == 0:
        diagram_title += '\nTraining does not consider speaker info'
    elif args.speaker_info_train == 1:
        diagram_title += '\nTraining considers intra-speaker only'
    elif args.speaker_info_train == 2:
        diagram_title += '\nTraining considers intra-speaker & inter-speaker'

    emo_mapping_dict1 = {'a':'ang', 'h':'hap', 'n':'neu', 's':'sad', 'S':'Start', 'd':'End', 'p':'pre-trained'}
    emo_mapping_dict2 = {'ang':'a', 'hap':'h', 'neu':'n', 'sad':'s', 'Start':'Start', 'End':'End', 'pre-trained':'p'}
    emo_index_dict = {'a':0, 'h':1, 'n':2, 's':3, 'ang':0, 'hap':1, 'neu':2, 'sad':3}

    if args.dataset == 'Original':
        emo_dict = joblib.load('./data/emo_all_iemocap.pkl')
        dialogs_edit = joblib.load('./data/dialog_iemocap.pkl')
    elif args.dataset == 'C2C':
        emo_dict = joblib.load('./data/C2C_4emo_all_iemocap.pkl')
    elif args.dataset == 'U2U':
        emo_dict = joblib.load('./data/U2U_4emo_all_iemocap.pkl')
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    out_dict = joblib.load('./data/outputs.pkl')

    Ses_01_X = []
    Ses_02_X = []
    Ses_03_X = []
    Ses_04_X = []
    Ses_05_X = []

    Ses_01_Y = []
    Ses_02_Y = []
    Ses_03_Y = []
    Ses_04_Y = []
    Ses_05_Y = []

    X = {'Ses01':[], 'Ses02':[], 'Ses03':[], 'Ses04':[], 'Ses05':[]} #observed utterance
    Y = {'Ses01':[], 'Ses02':[], 'Ses03':[], 'Ses04':[], 'Ses05':[]} #observed emotion(only record ang, hap, neu, sad)
    for dialog in dialogs.values():
        for utt in dialog:
            if emo_dict[utt] == 'ang' or emo_dict[utt] == 'hap' or emo_dict[utt] == 'neu' or emo_dict[utt] == 'sad':
                Session_num = utt[0:5]
                if Session_num == 'Ses01':
                    Ses_01_X.append(utt)
                    Ses_01_Y.append(emo_dict[utt])
                elif Session_num == 'Ses02':
                    Ses_02_X.append(utt)
                    Ses_02_Y.append(emo_dict[utt])
                elif Session_num == 'Ses03':
                    Ses_03_X.append(utt)
                    Ses_03_Y.append(emo_dict[utt])
                elif Session_num == 'Ses04':
                    Ses_04_X.append(utt)
                    Ses_04_Y.append(emo_dict[utt])
                elif Session_num == 'Ses05':
                    Ses_05_X.append(utt)
                    Ses_05_Y.append(emo_dict[utt])
            elif args.dataset == 'Original':
                dialogs_edit[utt[:-5]].remove(utt)

    X['Ses01'] = Ses_02_X + Ses_03_X + Ses_04_X + Ses_05_X
    X['Ses02'] = Ses_01_X + Ses_03_X + Ses_04_X + Ses_05_X
    X['Ses03'] = Ses_01_X + Ses_02_X + Ses_04_X + Ses_05_X
    X['Ses04'] = Ses_01_X + Ses_02_X + Ses_03_X + Ses_05_X
    X['Ses05'] = Ses_01_X + Ses_02_X + Ses_03_X + Ses_04_X

    Y['Ses01'] = Ses_02_Y + Ses_03_Y + Ses_04_Y + Ses_05_Y
    Y['Ses02'] = Ses_01_Y + Ses_03_Y + Ses_04_Y + Ses_05_Y
    Y['Ses03'] = Ses_01_Y + Ses_02_Y + Ses_04_Y + Ses_05_Y
    Y['Ses04'] = Ses_01_Y + Ses_02_Y + Ses_03_Y + Ses_05_Y
    Y['Ses05'] = Ses_01_Y + Ses_02_Y + Ses_03_Y + Ses_04_Y

    # trans_prob = utils.emo_trans_prob_BI_without_softmax(emo_dict, dialogs)
    if args.dataset == 'Original':
        no_speaker_info_emo_trans_prob_dict, intra_emo_trans_prob_dict, inter_emo_trans_prob_dict = utils.get_val_emo_trans_prob(emo_dict, dialogs_edit)
    else:
        no_speaker_info_emo_trans_prob_dict, intra_emo_trans_prob_dict, inter_emo_trans_prob_dict = utils.get_val_emo_trans_prob(emo_dict, dialogs)
    '''
    # pre-trained calssifier中增加8項，以logits計算
    out_dict['Start2a'] = math.log(trans_prob['Start2a']/(1-trans_prob['Start2a']), math.e)
    out_dict['Start2h'] = math.log(trans_prob['Start2h']/(1-trans_prob['Start2h']), math.e)
    out_dict['Start2n'] = math.log(trans_prob['Start2n']/(1-trans_prob['Start2n']), math.e)
    out_dict['Start2s'] = math.log(trans_prob['Start2s']/(1-trans_prob['Start2s']), math.e)
    
    out_dict['a2End'] = 10000 #log(無限大)
    out_dict['h2End'] = 10000
    out_dict['n2End'] = 10000
    out_dict['s2End'] = 10000
    '''

    Gau_rand_num = np.random.normal(0, 0.000001, 28) #0-mean gaussian distribution
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
    Index = 0
    for weight_name in W:
        W[weight_name] = Gau_rand_num[Index]
        Index += 1

    # object init
    if args.speaker_info_train == 0:
        CRF_model_Ses01 = CRF_SGD(W.copy(), X['Ses01'], Y['Ses01'], no_speaker_info_emo_trans_prob_dict['Ses01'], {}, {}, out_dict, args.learning_rate)
        CRF_model_Ses02 = CRF_SGD(W.copy(), X['Ses02'], Y['Ses02'], no_speaker_info_emo_trans_prob_dict['Ses02'], {}, {}, out_dict, args.learning_rate)
        CRF_model_Ses03 = CRF_SGD(W.copy(), X['Ses03'], Y['Ses03'], no_speaker_info_emo_trans_prob_dict['Ses03'], {}, {}, out_dict, args.learning_rate)
        CRF_model_Ses04 = CRF_SGD(W.copy(), X['Ses04'], Y['Ses04'], no_speaker_info_emo_trans_prob_dict['Ses04'], {}, {}, out_dict, args.learning_rate)
        CRF_model_Ses05 = CRF_SGD(W.copy(), X['Ses05'], Y['Ses05'], no_speaker_info_emo_trans_prob_dict['Ses05'], {}, {}, out_dict, args.learning_rate)
    else:
        CRF_model_Ses01 = CRF_SGD(W.copy(), X['Ses01'], Y['Ses01'], {}, inter_emo_trans_prob_dict['Ses01'], intra_emo_trans_prob_dict['Ses01'], out_dict, args.learning_rate)
        CRF_model_Ses02 = CRF_SGD(W.copy(), X['Ses02'], Y['Ses02'], {}, inter_emo_trans_prob_dict['Ses02'], intra_emo_trans_prob_dict['Ses02'], out_dict, args.learning_rate)
        CRF_model_Ses03 = CRF_SGD(W.copy(), X['Ses03'], Y['Ses03'], {}, inter_emo_trans_prob_dict['Ses03'], intra_emo_trans_prob_dict['Ses03'], out_dict, args.learning_rate)
        CRF_model_Ses04 = CRF_SGD(W.copy(), X['Ses04'], Y['Ses04'], {}, inter_emo_trans_prob_dict['Ses04'], intra_emo_trans_prob_dict['Ses04'], out_dict, args.learning_rate)
        CRF_model_Ses05 = CRF_SGD(W.copy(), X['Ses05'], Y['Ses05'], {}, inter_emo_trans_prob_dict['Ses05'], intra_emo_trans_prob_dict['Ses05'], out_dict, args.learning_rate)

    emo_dict_label = joblib.load('./data/emo_all_iemocap.pkl')
    label = []
    for _, dia in enumerate(dialogs):
        label += [utils.convert_to_index(emo_dict_label[utt]) for utt in dialogs[dia]]

    plt.figure()
    ann_list = []
    plt.axis([0, args.iteration, 0.5, 0.8])
    
    iters = [0, 0]
    uars = [0.5, 0.5]
    accs = [0.5, 0.5]
    uars_arr = np.zeros(shape=(1,0))
    accs_arr = np.zeros(shape=(1,0))

    for Iter in range(1, args.iteration + 1, 1):
        print('training iteration : '+str(Iter)+'/'+str(args.iteration))
        CRF_model_Ses01.update()
        CRF_model_Ses02.update()
        CRF_model_Ses03.update()
        CRF_model_Ses04.update()
        CRF_model_Ses05.update()
        uar, acc, conf = test_acc(CRF_model_Ses01.W, CRF_model_Ses02.W, CRF_model_Ses03.W, CRF_model_Ses04.W, CRF_model_Ses05.W)
        uar = round(uar, 3)
        acc = round(acc, 3)
        plot_dynamic_line_chart(uars, accs, Iter, args.iteration, uar, acc)
        print('==========================================')
    plt.savefig('result/uar&acc.png')

    plt.figure()
    sn.heatmap(conf, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('DED performance: uar: %.3f, acc: %.3f\n%s' % (uar, acc, diagram_title), fontsize=8)
    plt.savefig('result/confusion_matrix.png')
    plt.show()

    file1=open('weight/Ses01_weight.pickle','wb')
    file2=open('weight/Ses02_weight.pickle','wb')
    file3=open('weight/Ses03_weight.pickle','wb')
    file4=open('weight/Ses04_weight.pickle','wb')
    file5=open('weight/Ses05_weight.pickle','wb')

    pickle.dump(CRF_model_Ses01.W, file1)
    pickle.dump(CRF_model_Ses02.W, file2)
    pickle.dump(CRF_model_Ses03.W, file3)
    pickle.dump(CRF_model_Ses04.W, file4)
    pickle.dump(CRF_model_Ses05.W, file5)

    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    
    print("====================args====================")
    print(args.iteration, 'iteration')
    print('learning rate:', args.learning_rate)
    print(args.dataset, 'dataset')
    if args.concatenation == 1:
        print('With concatenation')
    else:
        print('Without concatenation')
    print('Viterbi algo. with', args.inter_intra_test)
    if args.speaker_info_train == 0:
        print('Training does not consider speaker info')
    elif args.speaker_info_train == 1:
        print('Training considers intra-speaker only')
    elif args.speaker_info_train == 2:
        print('Training considers intra-speaker & inter-speaker')
    print("====================args====================")