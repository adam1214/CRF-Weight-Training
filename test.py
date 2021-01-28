import pickle
import joblib
import utils
import math

def viterbi(Weight, dialogs):
    emo_list = ['a', 'h', 'n', 's']
    predict = []
    Q = [([0]*4) for i in range(len(dialogs))] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    
    # 第一個時間點 (f_j*w_j)
    Q[0][0] = math.exp(Weight['Start2a']*1 + Weight['p_a']*out_dict[dialogs[0]][0])
    Q[0][1] = math.exp(Weight['Start2h']*1 + Weight['p_h']*out_dict[dialogs[0]][1])
    Q[0][2] = math.exp(Weight['Start2n']*1 + Weight['p_n']*out_dict[dialogs[0]][2])
    Q[0][3] = math.exp(Weight['Start2s']*1 + Weight['p_s']*out_dict[dialogs[0]][3])
    emo_vals = [Q[0][0], Q[0][1], Q[0][2], Q[0][3]]
    max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
    predict.append(max_index)

    for i in range(1, len(dialogs), 1):
        for j in range(0, 4, 1): # j = 0,1,2,3
            candi_0 = Q[i-1][0] * math.exp(Weight[emo_list[0]+'2'+emo_list[j]]*1 + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][0])
            candi_1 = Q[i-1][1] * math.exp(Weight[emo_list[1]+'2'+emo_list[j]]*1 + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][1])
            candi_2 = Q[i-1][2] * math.exp(Weight[emo_list[2]+'2'+emo_list[j]]*1 + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][2])
            candi_3 = Q[i-1][3] * math.exp(Weight[emo_list[3]+'2'+emo_list[j]]*1 + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][3])
            
            candi_list = [candi_0, candi_1, candi_2, candi_3]
            max_val =  max(candi_list) # 返回最大值
            Q[i][j] = max_val
        emo_vals = [Q[i][0], Q[i][1], Q[i][2], Q[i][3]]
        max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
        predict.append(max_index)
    #print(Q)
    return predict

if __name__ == "__main__":
    with open('weight.pickle','rb') as file:
        Weight = pickle.load(file)
    # print(Weight)
    # print(viterbi(Weight, 30))
    
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    emo_dict = joblib.load('./data/emo_all_iemocap.pkl')
    out_dict = joblib.load('./data/outputs.pkl')
    
    predict = []
    label = []
    for i, dia in enumerate(dialogs):
        print("Decoding dialog: {}/{}, {}".format(i+1,len(dialogs),dia))
        #print(dia)
        #print(len(dialogs[dia]))
        label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]
        predict += viterbi(Weight, dialogs[dia])
            
        
    
    uar, acc, conf = utils.evaluate(predict, label)
    print('DED performance: uar: %.3f, acc: %.3f' % (uar, acc))
    print('Confusion matrix:\n%s' % conf)
    