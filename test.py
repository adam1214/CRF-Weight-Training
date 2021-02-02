import itertools

emo_com = itertools.product(['ang', 'hap', 'neu', 'sad'], repeat = 2)   
emo_com_list = [item for item in emo_com] 
Start_emo_com_list = [('Start', 'ang'), ('Start', 'hap'), ('Start', 'neu'), ('Start', 'sad')]
for t in range(1,4,1):
    if t == 1:
        tmp_emo_com_list = Start_emo_com_list #len is 4 (c = 0~3)
        c = 0
    else:
        tmp_emo_com_list = emo_com_list #len is 16 (c = 4~19)
        c = 4
    for emo_com_item in tmp_emo_com_list:
        print(emo_com_item)
    print('===========')
