import pickle
import joblib
import utils
import math
import seaborn as sn
import matplotlib.pyplot as plt

def viterbi_inter(Weight, dialogs, no_speaker_info_emo_trans_prob_dict, inter_emo_trans_prob_dict, intra_emo_trans_prob_dict, out_dict, concatenate_or_not, speaker_info_train, validation_or_test):
    emo_list = ['a', 'h', 'n', 's']
    if validation_or_test == 'test':
        predict = []
    else:
        predict_dict = {}
    trans_prob = {}
    Q = [([0]*4) for i in range(len(dialogs))] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    
    # 第一個時間點 (f_j*w_j)
    if speaker_info_train == 0:
        trans_prob = no_speaker_info_emo_trans_prob_dict
    else:
        trans_prob = intra_emo_trans_prob_dict
    Q[0][0] = Weight['Start2a']*trans_prob['Start2a'] + Weight['p_a']*out_dict[dialogs[0]][0]
    Q[0][1] = Weight['Start2h']*trans_prob['Start2h'] + Weight['p_h']*out_dict[dialogs[0]][1]
    Q[0][2] = Weight['Start2n']*trans_prob['Start2n'] + Weight['p_n']*out_dict[dialogs[0]][2]
    Q[0][3] = Weight['Start2s']*trans_prob['Start2s'] + Weight['p_s']*out_dict[dialogs[0]][3]
    emo_vals = [Q[0][0], Q[0][1], Q[0][2], Q[0][3]]
    max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
    if validation_or_test == 'test':
        predict.append(max_index)
    else:
        predict_dict[dialogs[0]] = max_index

    for i in range(1, len(dialogs), 1):
        pre_utt = dialogs[i-1]
        cur_utt = dialogs[i]
        for j in range(0, 4, 1): # j = 0,1,2,3
            if speaker_info_train == 0:
                trans_prob = no_speaker_info_emo_trans_prob_dict
            elif pre_utt[-4] == cur_utt[-4]:
                trans_prob = intra_emo_trans_prob_dict
            else:
                trans_prob = inter_emo_trans_prob_dict
            candi_0 = Q[i-1][0] + Weight[emo_list[0]+'2'+emo_list[j]]*trans_prob[emo_list[0]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j]
            candi_1 = Q[i-1][1] + Weight[emo_list[1]+'2'+emo_list[j]]*trans_prob[emo_list[1]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j]
            candi_2 = Q[i-1][2] + Weight[emo_list[2]+'2'+emo_list[j]]*trans_prob[emo_list[2]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j]
            candi_3 = Q[i-1][3] + Weight[emo_list[3]+'2'+emo_list[j]]*trans_prob[emo_list[3]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[dialogs[i]][j]
            
            candi_list = [candi_0, candi_1, candi_2, candi_3]
            max_val =  max(candi_list) # 返回最大值
            Q[i][j] = max_val
        emo_vals = [Q[i][0], Q[i][1], Q[i][2], Q[i][3]]
        max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
        if validation_or_test == 'test':
            predict.append(max_index)
        else:
            predict_dict[dialogs[i]] = max_index
    if concatenate_or_not == 0:
        if validation_or_test == 'test':
            return predict
        else:
            return predict_dict
    else:
        if validation_or_test == 'test':
            return predict[int(len(predict)/2):len(predict)]
        else:
            return predict_dict

def viterbi_intra(Weight, dialogs, no_speaker_info_emo_trans_prob_dict, intra_emo_trans_prob_dict, out_dict, concatenate_or_not, speaker_info_train, validation_or_test): # better than viterbi_inter
    emo_list = ['a', 'h', 'n', 's']
    M_utts = []
    F_utts = []
    for utt in dialogs:
        if utt[-4] == 'M':
            M_utts.append(utt)
        else:
            F_utts.append(utt)
    
    speakers_utts = [M_utts, F_utts]
    predict_dict = {}
    for speaker_utts in speakers_utts:
        Q = [([0]*4) for i in range(len(speaker_utts))] # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        
        # 第一個時間點 (f_j*w_j)
        if speaker_info_train == 0:
            trans_prob = no_speaker_info_emo_trans_prob_dict
        else:
            trans_prob = intra_emo_trans_prob_dict
        Q[0][0] = Weight['Start2a']*trans_prob['Start2a'] + Weight['p_a']*out_dict[speaker_utts[0]][0]
        Q[0][1] = Weight['Start2h']*trans_prob['Start2h'] + Weight['p_h']*out_dict[speaker_utts[0]][1]
        Q[0][2] = Weight['Start2n']*trans_prob['Start2n'] + Weight['p_n']*out_dict[speaker_utts[0]][2]
        Q[0][3] = Weight['Start2s']*trans_prob['Start2s'] + Weight['p_s']*out_dict[speaker_utts[0]][3]
        emo_vals = [Q[0][0], Q[0][1], Q[0][2], Q[0][3]]
        max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
        predict_dict[speaker_utts[0]] = max_index

        for i in range(1, len(speaker_utts), 1):
            for j in range(0, 4, 1): # j = 0,1,2,3
                if speaker_info_train == 0:
                    trans_prob = no_speaker_info_emo_trans_prob_dict
                else:
                    trans_prob = intra_emo_trans_prob_dict
                candi_0 = Q[i-1][0] + Weight[emo_list[0]+'2'+emo_list[j]]*trans_prob[emo_list[0]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[speaker_utts[i]][j]
                candi_1 = Q[i-1][1] + Weight[emo_list[1]+'2'+emo_list[j]]*trans_prob[emo_list[1]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[speaker_utts[i]][j]
                candi_2 = Q[i-1][2] + Weight[emo_list[2]+'2'+emo_list[j]]*trans_prob[emo_list[2]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[speaker_utts[i]][j]
                candi_3 = Q[i-1][3] + Weight[emo_list[3]+'2'+emo_list[j]]*trans_prob[emo_list[3]+'2'+emo_list[j]] + Weight['p_'+emo_list[j]]*out_dict[speaker_utts[i]][j]
                
                candi_list = [candi_0, candi_1, candi_2, candi_3]
                max_val =  max(candi_list) # 返回最大值
                Q[i][j] = max_val
            emo_vals = [Q[i][0], Q[i][1], Q[i][2], Q[i][3]]
            max_index = emo_vals.index(max(emo_vals)) # 最大值的索引
            predict_dict[speaker_utts[i]] = max_index
    predict = []
    for utt in dialogs:
        predict.append(predict_dict[utt])

    if concatenate_or_not == 0:
        if validation_or_test == 'test':
            return predict
        else:
            return predict_dict
    else:
        if validation_or_test == 'test':
            return predict[int(len(predict)/2):len(predict)]
        else:
            return predict_dict

if __name__ == "__main__":
    '''
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
    '''