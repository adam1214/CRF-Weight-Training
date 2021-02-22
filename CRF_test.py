import pickle
import joblib
import utils
import math
import seaborn as sn
import matplotlib.pyplot as plt

def viterbi(Weight, dialogs, trans_prob, out_dict):
    emo_list = ['a', 'h', 'n', 's']
    predict = []
    Q = [([0]*4) for i in range(len(dialogs))] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    
    # 第一個時間點 (f_j*w_j)
    Q[0][0] = math.exp(Weight['Start2a']*trans_prob['Start2a'] + Weight['p_a']*out_dict[dialogs[0]][0])
    Q[0][1] = math.exp(Weight['Start2h']*trans_prob['Start2h'] + Weight['p_h']*out_dict[dialogs[0]][1])
    Q[0][2] = math.exp(Weight['Start2n']*trans_prob['Start2n'] + Weight['p_n']*out_dict[dialogs[0]][2])
    Q[0][3] = math.exp(Weight['Start2s']*trans_prob['Start2s'] + Weight['p_s']*out_dict[dialogs[0]][3])
    emo_vals = [Q[0][0], Q[0][1], Q[0][2], Q[0][3]]
    max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
    predict.append(max_index)

    for i in range(1, len(dialogs), 1):
        for j in range(0, 4, 1): # j = 0,1,2,3
            candi_0 = Q[i-1][0] * math.exp(Weight[emo_list[0]+'2'+emo_list[j]]*trans_prob[emo_list[0]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j])
            candi_1 = Q[i-1][1] * math.exp(Weight[emo_list[1]+'2'+emo_list[j]]*trans_prob[emo_list[1]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j])
            candi_2 = Q[i-1][2] * math.exp(Weight[emo_list[2]+'2'+emo_list[j]]*trans_prob[emo_list[2]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j])
            candi_3 = Q[i-1][3] * math.exp(Weight[emo_list[3]+'2'+emo_list[j]]*trans_prob[emo_list[3]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j])
            
            candi_list = [candi_0, candi_1, candi_2, candi_3]
            max_val =  max(candi_list) # 返回最大值
            Q[i][j] = max_val
        emo_vals = [Q[i][0], Q[i][1], Q[i][2], Q[i][3]]
        max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
        predict.append(max_index)
    #print(Q)
    return predict

if __name__ == "__main__":
    with open('weight/Ses01_weight.pickle','rb') as file:
        S1_Weight = pickle.load(file)
    with open('weight/Ses02_weight.pickle','rb') as file:
        S2_Weight = pickle.load(file)
    with open('weight/Ses03_weight.pickle','rb') as file:
        S3_Weight = pickle.load(file)
    with open('weight/Ses04_weight.pickle','rb') as file:
        S4_Weight = pickle.load(file)
    with open('weight/Ses05_weight.pickle','rb') as file:
        S5_Weight = pickle.load(file)
    
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    emo_dict = joblib.load('./data/emo_all_iemocap.pkl')
    out_dict = joblib.load('./data/outputs.pkl')

    #trans_prob = utils.emo_trans_prob_BI_without_softmax(joblib.load('./data/U2U_4emo_all_iemmcap.pkl'), dialogs)
    val_emo_trans_prob = utils.get_val_emo_trans_prob(joblib.load('./data/U2U_4emo_all_iemocap.pkl'), dialogs)
    
    label = []
    predict = []
    for i, dia in enumerate(dialogs):
        #print("Decoding dialog: {}/{}, {}".format(i+1,len(dialogs),dia))
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
        predict += viterbi(W, dialogs[dia], val_emo_trans_prob[Session_num], out_dict)
        label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]

    uar, acc, conf = utils.evaluate(predict, label)
    print('DED performance: uar: %.3f, acc: %.3f' % (uar, acc))
    print(conf)
    sn.heatmap(conf, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('DED performance: uar: %.3f, acc: %.3f' % (uar, acc))
    plt.savefig('result/confusion_matrix.png')
    plt.show()
