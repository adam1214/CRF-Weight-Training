import pickle
def gen_feature_vec():
    '''
    feature vector dims is 24:

    Start2a Start2h Start2n Start2s
    a2a     a2h     a2n     a2s
    h2a     h2h     h2n     h2s
    n2a     n2h     n2n     n2s
    s2a     s2h     s2n     s2s
    a2End   h2End   n2End   s2End
    '''
    feature_vec_list = []
    #Initial state, [Start, a] and [Start, h] and [Start, n] and [Start, s]
    feature_vec_list.append({'Start_a':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'Start_h':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'Start_n':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            'Start_s':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]})
    
    for dialog_len in range(2, 111, 1): #max dialog length is 110
        print("Processing dialog len", dialog_len)
        feature_vec_dict = {}
        for pre_utt in feature_vec_list[dialog_len-2].keys():
            #print(pre_utt[-1])
            #print(feature_vec_list[dialog_len-2][pre_utt])
            if pre_utt[-1] == 'a':
                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[4] += 1
                feature_vec_dict[pre_utt+'_a'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[5] += 1
                feature_vec_dict[pre_utt+'_h'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[6] += 1
                feature_vec_dict[pre_utt+'_n'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[7] += 1
                feature_vec_dict[pre_utt+'_s'] = list_copy

            elif pre_utt[-1] == 'h':
                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[8] += 1
                feature_vec_dict[pre_utt+'_a'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[9] += 1
                feature_vec_dict[pre_utt+'_h'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[10] += 1
                feature_vec_dict[pre_utt+'_n'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[11] += 1
                feature_vec_dict[pre_utt+'_s'] = list_copy

            elif pre_utt[-1] == 'n':
                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[12] += 1
                feature_vec_dict[pre_utt+'_a'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[13] += 1
                feature_vec_dict[pre_utt+'_h'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[14] += 1
                feature_vec_dict[pre_utt+'_n'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[15] += 1
                feature_vec_dict[pre_utt+'_s'] = list_copy

            elif pre_utt[-1] == 's':
                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[16] += 1
                feature_vec_dict[pre_utt+'_a'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[17] += 1
                feature_vec_dict[pre_utt+'_h'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[18] += 1
                feature_vec_dict[pre_utt+'_n'] = list_copy

                list_copy = feature_vec_list[dialog_len-2][pre_utt].copy()
                list_copy[19] += 1
                feature_vec_dict[pre_utt+'_s'] = list_copy

        feature_vec_list.append(feature_vec_dict)
    '''
    for dic in feature_vec_list:
        print(len(dic))
        for d in dic:
            print(d, dic[d])
    '''
    with open('feature_vec_list', 'wb') as fp:
        pickle.dump(feature_vec_list, fp)
if __name__ == "__main__":
    gen_feature_vec()
    feature_vec_list = []
    with open ('feature_vec_list', 'rb') as fp:
        feature_vec_list = pickle.load(fp)

    for dic in feature_vec_list:
        print(len(dic))
        for d in dic:
            print(d, dic[d])
    