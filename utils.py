import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
import numpy as np

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
  label, predict = label[index], predict[index]

  return recall_score(label, predict, average='macro'), accuracy_score(label, predict), confusion_matrix(label, predict)

def emo_trans_prob_BI_without_softmax(emo_dict, dialogs, val=None):
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

    pre_emo = ''
    pre_dialog_id = ''

    for dialog in dialogs.values():
        utt_num = 1
        for utt in dialog:
            dialog_id = utt[0:-5]
            #print(dialog_id)
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

            if emo_dict[utt] != 'ang' and emo_dict[utt] != 'hap' and emo_dict[utt] != 'neu' and emo_dict[utt] != 'sad': 
                # only estimate anger, happiness, neutral, sadness
                pre_dialog_id = dialog_id
                continue

            if pre_emo == '' and pre_dialog_id == '': # begining of the traversal
                pre_emo = emo_dict[utt]
                pre_dialog_id = dialog_id
                continue

            if pre_dialog_id != dialog_id: # new dialog
                pre_emo = ''

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
            
            pre_dialog_id = dialog_id
            pre_emo = emo_dict[utt]
            
            utt_num += 1
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

if __name__ == "__main__":
    pass
    '''
    emo_dict = joblib.load('./data/U2U_4emo_all_iemmcap.pkl')
    dialogs = joblib.load('./data/dialog_iemocap.pkl')
    trans_prob = emo_trans_prob_BI_without_softmax(emo_dict, dialogs)
    '''
    