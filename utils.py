import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
import numpy as np
import random

def convert_to_index(emotion):
    """convert emotion to index """
    map_emo = {'ang':0, 'hap':1, 'neu':2, 'sad':3}
    if emotion in map_emo.keys():
        return map_emo[emotion]
    else:
        return -1

def evaluate(predict, label): #uar, acc, conf = utils.evaluate(predict, label)
    # Only evaluate utterances labeled in defined 4 emotion states
    label, predict = np.array(label), np.array(predict)
    index = [label != -1]
    label, predict = label[tuple(index)], predict[tuple(index)]

    return recall_score(label, predict, average='macro'), accuracy_score(label, predict), confusion_matrix(label, predict)

def split_dialog(dialogs):
    """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
        See eq (5) in the paper.
    Arg:
        dialogs: dict, for example, utterances of two speakers in dialog_01: 
                {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
    Return:
        spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
                {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
                dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
    """

    spk_dialogs = {}
    for dialog_id in dialogs.keys():
        spk_dialogs[dialog_id+'_M'] = []
        spk_dialogs[dialog_id+'_F'] = []
        for utt_id in dialogs[dialog_id]:
            if utt_id[-4] == 'M':
                spk_dialogs[dialog_id+'_M'].append(utt_id)
            elif utt_id[-4] == 'F':
                spk_dialogs[dialog_id+'_F'].append(utt_id)

    return spk_dialogs

def emo_trans_prob_BI_without_softmax_inter(emo_dict, dialogs, val=None):
    # only estimate anger, happiness, neutral, sadness
    a2 = 0
    h2 = 0
    n2 = 0
    s2 = 0

    ang2ang = 0
    ang2hap = 0
    ang2neu = 0
    ang2sad = 0
    
    hap2ang = 0
    hap2hap = 0
    hap2neu = 0
    hap2sad = 0

    neu2ang = 0
    neu2hap = 0
    neu2neu = 0
    neu2sad = 0

    sad2ang = 0
    sad2hap = 0
    sad2neu = 0
    sad2sad = 0

    for dialog in dialogs.values():
        pre_emo = ''
        pre_spk_id = ''
        for utt in dialog:
            dialog_id = utt[0:-5]
            if val and val == dialog_id[0:5]:
                continue
            if pre_emo == '': # begining of the traversal
                pre_emo = emo_dict[utt]
                pre_spk_id = utt[-4]
                continue
            if pre_spk_id != utt[-4]:
                if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                    ang2ang += 1
                    a2 += 1
                if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                    ang2hap += 1
                    a2 += 1
                if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                    ang2neu += 1
                    a2 += 1
                if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                    ang2sad += 1
                    a2 += 1

                if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                    hap2ang += 1
                    h2 += 1
                if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                    hap2hap += 1
                    h2 += 1
                if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                    hap2neu += 1
                    h2 += 1
                if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                    hap2sad += 1
                    h2 += 1

                if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                    neu2ang += 1
                    n2 += 1
                if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                    neu2hap += 1
                    n2 += 1
                if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                    neu2neu += 1
                    n2 += 1
                if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                    neu2sad += 1
                    n2 += 1

                if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                    sad2ang += 1
                    s2 += 1
                if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                    sad2hap += 1
                    s2 += 1
                if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                    sad2neu += 1
                    s2 += 1
                if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                    sad2sad += 1
                    s2 += 1
            
            pre_emo = emo_dict[utt]
    '''
    print(ang2ang/a2+ang2hap/a2+ang2neu/a2+ang2sad/a2)
    print(hap2ang/h2+hap2hap/h2+hap2neu/h2+hap2sad/h2)
    print(neu2ang/n2+neu2hap/n2+neu2neu/n2+neu2sad/n2)
    print(sad2ang/s2+sad2hap/s2+sad2neu/s2+sad2sad/s2)
    print('=============================================')
    '''
    return {'a2a':ang2ang/a2, 'a2h':ang2hap/a2, 'a2n':ang2neu/a2, 'a2s':ang2sad/a2, \
            'h2a':hap2ang/h2, 'h2h':hap2hap/h2, 'h2n':hap2neu/h2, 'h2s':hap2sad/h2, \
            'n2a':neu2ang/n2, 'n2h':neu2hap/n2, 'n2n':neu2neu/n2, 'n2s':neu2sad/n2, \
            's2a':sad2ang/s2, 's2h':sad2hap/s2, 's2n':sad2neu/s2, 's2s':sad2sad/s2  }
 
def emo_trans_prob_BI_without_softmax_intra(emo_dict, dialogs, val=None):
    # only estimate anger, happiness, neutral, sadness
    Start2a = 0
    Start2h = 0
    Start2n = 0
    Start2s = 0
    
    a2End = 0
    h2End = 0
    n2End = 0
    s2End = 0
    
    Start2 = 0
    End2 = 0
    a2 = 0
    h2 = 0
    n2 = 0
    s2 = 0

    ang2ang = 0
    ang2hap = 0
    ang2neu = 0
    ang2sad = 0
    
    hap2ang = 0
    hap2hap = 0
    hap2neu = 0
    hap2sad = 0

    neu2ang = 0
    neu2hap = 0
    neu2neu = 0
    neu2sad = 0

    sad2ang = 0
    sad2hap = 0
    sad2neu = 0
    sad2sad = 0

    spk_dialogs = split_dialog(dialogs)
    for utt_list in spk_dialogs.values():
        pre_emo = ''
        utt_num = 1
        for utt in utt_list:
            dialog_id = utt[0:-5]
            if val and val == dialog_id[0:5]:
                continue
            if utt_num == 1:
                Start2 += 1
                if emo_dict[utt] == 'ang':
                    Start2a += 1
                elif emo_dict[utt] == 'hap':
                    Start2h += 1
                elif emo_dict[utt] == 'neu':
                    Start2n += 1
                elif emo_dict[utt] == 'sad':
                    Start2s += 1

            if utt_num == len(utt_list):
                End2 += 1
                if emo_dict[utt] == 'ang':
                    a2End += 1
                elif emo_dict[utt] == 'hap':
                    h2End += 1
                elif emo_dict[utt] == 'neu':
                    n2End += 1
                elif emo_dict[utt] == 'sad':
                    s2End += 1
            utt_num += 1
            
            if pre_emo == '': # begining of the dialog
                pre_emo = emo_dict[utt]
                continue

            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                ang2ang += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                ang2hap += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                ang2neu += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                ang2sad += 1
                a2 += 1

            if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hap2ang += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hap2hap += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hap2neu += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hap2sad += 1
                h2 += 1

            if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                neu2ang += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                neu2hap += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                neu2neu += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                neu2sad += 1
                n2 += 1

            if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                sad2ang += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                sad2hap += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                sad2neu += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sad2sad += 1
                s2 += 1
            
            pre_emo = emo_dict[utt]
    '''
    print(ang2ang/a2+ang2hap/a2+ang2neu/a2+ang2sad/a2)
    print(hap2ang/h2+hap2hap/h2+hap2neu/h2+hap2sad/h2)
    print(neu2ang/n2+neu2hap/n2+neu2neu/n2+neu2sad/n2)
    print(sad2ang/s2+sad2hap/s2+sad2neu/s2+sad2sad/s2)
    print((Start2a+Start2h+Start2n+Start2s)/Start2)
    print((a2End+h2End+n2End+s2End)/End2)
    print('=============================================')
    '''
    return {'a2a':ang2ang/a2, 'a2h':ang2hap/a2, 'a2n':ang2neu/a2, 'a2s':ang2sad/a2, \
            'h2a':hap2ang/h2, 'h2h':hap2hap/h2, 'h2n':hap2neu/h2, 'h2s':hap2sad/h2, \
            'n2a':neu2ang/n2, 'n2h':neu2hap/n2, 'n2n':neu2neu/n2, 'n2s':neu2sad/n2, \
            's2a':sad2ang/s2, 's2h':sad2hap/s2, 's2n':sad2neu/s2, 's2s':sad2sad/s2, \
            'Start2a':Start2a/Start2, 'Start2h':Start2h/Start2, 'Start2n':Start2n/Start2, 'Start2s':Start2s/Start2, \
            'a2End':a2End/End2, 'h2End':h2End/End2, 'n2End':n2End/End2, 's2End':s2End/End2 }

def emo_trans_prob_BI_without_softmax_no_speaker_info(emo_dict, dialogs, val=None):
    # only estimate anger, happiness, neutral, sadness
    Start2a = 0
    Start2h = 0
    Start2n = 0
    Start2s = 0
    
    a2End = 0
    h2End = 0
    n2End = 0
    s2End = 0
    
    Start2 = 0
    End2 = 0
    a2 = 0
    h2 = 0
    n2 = 0
    s2 = 0

    ang2ang = 0
    ang2hap = 0
    ang2neu = 0
    ang2sad = 0
    
    hap2ang = 0
    hap2hap = 0
    hap2neu = 0
    hap2sad = 0

    neu2ang = 0
    neu2hap = 0
    neu2neu = 0
    neu2sad = 0

    sad2ang = 0
    sad2hap = 0
    sad2neu = 0
    sad2sad = 0

    for dialog in dialogs.values():
        utt_num = 1
        pre_emo = ''
        for utt in dialog:
            dialog_id = utt[0:-5]
            if val and val == dialog_id[0:5]:
                continue
            if utt_num == 1:
                Start2 += 1
                if emo_dict[utt] == 'ang':
                    Start2a += 1
                elif emo_dict[utt] == 'hap':
                    Start2h += 1
                elif emo_dict[utt] == 'neu':
                    Start2n += 1
                elif emo_dict[utt] == 'sad':
                    Start2s += 1

            if utt_num == len(dialog):
                End2 += 1
                if emo_dict[utt] == 'ang':
                    a2End += 1
                elif emo_dict[utt] == 'hap':
                    h2End += 1
                elif emo_dict[utt] == 'neu':
                    n2End += 1
                elif emo_dict[utt] == 'sad':
                    s2End += 1
            utt_num += 1

            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad': 
                # only estimate anger, happiness, neutral, sadness
                continue

            if pre_emo == '': # begining of the traversal
                pre_emo = emo_dict[utt]
                continue

            if pre_emo == 'ang' and emo_dict[utt] == 'ang':
                ang2ang += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'hap':
                ang2hap += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'neu':
                ang2neu += 1
                a2 += 1
            if pre_emo == 'ang' and emo_dict[utt] == 'sad':
                ang2sad += 1
                a2 += 1

            if pre_emo == 'hap' and emo_dict[utt] == 'ang':
                hap2ang += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'hap':
                hap2hap += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'neu':
                hap2neu += 1
                h2 += 1
            if pre_emo == 'hap' and emo_dict[utt] == 'sad':
                hap2sad += 1
                h2 += 1

            if pre_emo == 'neu' and emo_dict[utt] == 'ang':
                neu2ang += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'hap':
                neu2hap += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'neu':
                neu2neu += 1
                n2 += 1
            if pre_emo == 'neu' and emo_dict[utt] == 'sad':
                neu2sad += 1
                n2 += 1

            if pre_emo == 'sad' and emo_dict[utt] == 'ang':
                sad2ang += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'hap':
                sad2hap += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'neu':
                sad2neu += 1
                s2 += 1
            if pre_emo == 'sad' and emo_dict[utt] == 'sad':
                sad2sad += 1
                s2 += 1

            pre_emo = emo_dict[utt]
    '''
    print(ang2ang/a2+ang2hap/a2+ang2neu/a2+ang2sad/a2)
    print(hap2ang/h2+hap2hap/h2+hap2neu/h2+hap2sad/h2)
    print(neu2ang/n2+neu2hap/n2+neu2neu/n2+neu2sad/n2)
    print(sad2ang/s2+sad2hap/s2+sad2neu/s2+sad2sad/s2)
    print((Start2a+Start2h+Start2n+Start2s)/Start2)
    print((a2End+h2End+n2End+s2End)/End2)
    print('=============================================')
    '''
    return {'a2a':ang2ang/a2, 'a2h':ang2hap/a2, 'a2n':ang2neu/a2, 'a2s':ang2sad/a2, \
            'h2a':hap2ang/h2, 'h2h':hap2hap/h2, 'h2n':hap2neu/h2, 'h2s':hap2sad/h2, \
            'n2a':neu2ang/n2, 'n2h':neu2hap/n2, 'n2n':neu2neu/n2, 'n2s':neu2sad/n2, \
            's2a':sad2ang/s2, 's2h':sad2hap/s2, 's2n':sad2neu/s2, 's2s':sad2sad/s2, \
            'Start2a':Start2a/Start2, 'Start2h':Start2h/Start2, 'Start2n':Start2n/Start2, 'Start2s':Start2s/Start2, \
            'a2End':a2End/End2, 'h2End':h2End/End2, 'n2End':n2End/End2, 's2End':s2End/End2 }

def get_val_emo_trans_prob(emo_dict, dialogs):
    """Get emo_trans_prob estimated from training sessions."""

    session = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    no_speaker_info_emo_trans_prob_dict = {}
    intra_emo_trans_prob_dict = {}
    inter_emo_trans_prob_dict = {}
    for i in range(len(session)):
        val = session[i]

        no_speaker_info_emo_trans_prob_com = emo_trans_prob_BI_without_softmax_no_speaker_info(emo_dict, dialogs, val)
        no_speaker_info_emo_trans_prob_dict[val] = no_speaker_info_emo_trans_prob_com

        intra_emo_trans_prob_com = emo_trans_prob_BI_without_softmax_intra(emo_dict, dialogs, val)
        intra_emo_trans_prob_dict[val] = intra_emo_trans_prob_com

        inter_emo_trans_prob_com = emo_trans_prob_BI_without_softmax_inter(emo_dict, dialogs, val)
        inter_emo_trans_prob_dict[val] = inter_emo_trans_prob_com
    
    return no_speaker_info_emo_trans_prob_dict, intra_emo_trans_prob_dict, inter_emo_trans_prob_dict

def get_validation_sets(emo_dict, dialogs):
    random.seed(1)
    validation_dict = {'Ses01':{'ang':[],'hap':[],'neu':[],'sad':[]},'Ses02':{'ang':[],'hap':[],'neu':[],'sad':[]},'Ses03':{'ang':[],'hap':[],'neu':[],'sad':[]},'Ses04':{'ang':[],'hap':[],'neu':[],'sad':[]},'Ses05':{'ang':[],'hap':[],'neu':[],'sad':[]}}
    validation_dict_emos_utt = {'Ses01':[],'Ses02':[],'Ses03':[],'Ses04':[],'Ses05':[]}
    validation_dict_emos_label = {'Ses01':[],'Ses02':[],'Ses03':[],'Ses04':[],'Ses05':[]}
    session_utt_Quantity = {'Ses01':0,'Ses02':0,'Ses03':0,'Ses04':0,'Ses05':0}
    for utts_list in dialogs.values():
        session_utt_Quantity[(utts_list[0])[0:5]] += len(utts_list)
    #print(session_utt_Quantity)

    for Ses_num in validation_dict:
        for utts_list in dialogs.values():
            if (utts_list[0])[0:5] != Ses_num:
                continue
            else:
                for utt in utts_list:
                    if emo_dict[utt] == 'ang':
                        validation_dict[Ses_num]['ang'].append(utt)
                    elif emo_dict[utt] == 'hap':
                        validation_dict[Ses_num]['hap'].append(utt)
                    elif emo_dict[utt] == 'neu':
                        validation_dict[Ses_num]['neu'].append(utt)
                    elif emo_dict[utt] == 'sad':
                        validation_dict[Ses_num]['sad'].append(utt)
        validation_dict[Ses_num]['ang'] = random.sample(validation_dict[Ses_num]['ang'], int(session_utt_Quantity[Ses_num]*0.025))
        validation_dict[Ses_num]['hap'] = random.sample(validation_dict[Ses_num]['hap'], int(session_utt_Quantity[Ses_num]*0.025)) 
        validation_dict[Ses_num]['neu'] = random.sample(validation_dict[Ses_num]['neu'], int(session_utt_Quantity[Ses_num]*0.025)) 
        validation_dict[Ses_num]['sad'] = random.sample(validation_dict[Ses_num]['sad'], int(session_utt_Quantity[Ses_num]*0.025)) 

        validation_dict_emos_utt[Ses_num] = validation_dict[Ses_num]['ang'] + validation_dict[Ses_num]['hap'] + validation_dict[Ses_num]['neu'] + validation_dict[Ses_num]['sad']
        
        for utt in validation_dict_emos_utt[Ses_num]:
            if emo_dict[utt] == 'ang':
                validation_dict_emos_label[Ses_num].append(0)
            elif emo_dict[utt] == 'hap':
                validation_dict_emos_label[Ses_num].append(1)
            elif emo_dict[utt] == 'neu':
                validation_dict_emos_label[Ses_num].append(2)
            elif emo_dict[utt] == 'sad':
                validation_dict_emos_label[Ses_num].append(3)

    return validation_dict_emos_utt, validation_dict_emos_label

if __name__ == "__main__":
    
    emo_dict = joblib.load('./data/emo_all_iemocap.pkl')
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    
    validation_dict_emos_utt, validation_dict_emos_label = get_validation_sets(emo_dict, dialogs)
    #trans_prob = emo_trans_prob_BI_without_softmax_no_speaker_info(emo_dict, dialogs)
    #get_val_emo_trans_prob(emo_dict, dialogs)
    
    #intra_trans_probs = emo_trans_prob_BI_without_softmax_intra(emo_dict, dialogs)
    #inter_trans_probs = emo_trans_prob_BI_without_softmax_inter(emo_dict, dialogs)
    