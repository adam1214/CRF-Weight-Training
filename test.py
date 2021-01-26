import pickle
import joblib
import utils

def viterbi(Weight, T):
    emo_list = ['a', 'h', 'n', 's']
    predict = []
    Q = [([0]*4) for i in range(T)] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    
    # 第一個時間點
    Q[0][0] = Weight['Start2a'] * Weight['p_a']
    Q[0][1] = Weight['Start2h'] * Weight['p_h']
    Q[0][2] = Weight['Start2n'] * Weight['p_n']
    Q[0][3] = Weight['Start2s'] * Weight['p_s']
    emo_vals = [Q[0][0], Q[0][1], Q[0][2], Q[0][3]]
    max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
    predict.append(max_index)

    for i in range(1, T, 1):
        for j in range(0, 4, 1): # j = 0,1,2,3
            candi_0 = Q[i-1][0] * Weight[emo_list[0]+'2'+emo_list[j]] * Weight['p_'+emo_list[j]]
            candi_1 = Q[i-1][1] * Weight[emo_list[1]+'2'+emo_list[j]] * Weight['p_'+emo_list[j]]
            candi_2 = Q[i-1][2] * Weight[emo_list[2]+'2'+emo_list[j]] * Weight['p_'+emo_list[j]]
            candi_3 = Q[i-1][3] * Weight[emo_list[3]+'2'+emo_list[j]] * Weight['p_'+emo_list[j]]
            
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
    
    predict = []
    label = []
    for i, dia in enumerate(dialogs):
        print("Decoding dialog: {}/{}, {}".format(i+1,len(dialogs),dia))
        #print(dia)
        #print(len(dialogs[dia]))
        label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]
        predict += viterbi(Weight, len(dialogs[dia]))
        
    
    uar, acc, conf = utils.evaluate(predict, label)
    print('DED performance: uar: %.3f, acc: %.3f' % (uar, acc))
    print('Confusion matrix:\n%s' % conf)
    