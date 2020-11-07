import pickle
import numpy as np
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
    feature_vec_list.append({'Start_a_End':np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
                            'Start_h_End':np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
                            'Start_n_End':np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
                            'Start_s_End':np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])})
    
    for dialog_len in range(2, 111, 1): #max dialog length is 110
        print("Processing dialog len", dialog_len)
        feature_vec_dict = {}
        for pre_utt in feature_vec_list[dialog_len-2].keys():
            pre_utt_non_End = pre_utt[:-4]
            #print(pre_utt[-1])
            #print(feature_vec_list[dialog_len-2][pre_utt])
            if pre_utt_non_End[-1] == 'a':
                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[4] += 1
                num_arr_copy[20] = 1
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_a_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[5] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 1
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_h_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[6] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 1
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_n_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[7] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 1
                feature_vec_dict[pre_utt_non_End+'_s_End'] = num_arr_copy

            elif pre_utt_non_End[-1] == 'h':
                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[8] += 1
                num_arr_copy[20] = 1
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_a_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[9] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 1
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_h_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[10] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 1
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_n_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[11] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 1
                feature_vec_dict[pre_utt_non_End+'_s_End'] = num_arr_copy

            elif pre_utt_non_End[-1] == 'n':
                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[12] += 1
                num_arr_copy[20] = 1
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_a_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[13] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 1
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_h_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[14] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 1
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_n_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[15] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 1
                feature_vec_dict[pre_utt_non_End+'_s_End'] = num_arr_copy

            elif pre_utt_non_End[-1] == 's':
                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[16] += 1
                num_arr_copy[20] = 1
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_a_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[17] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 1
                num_arr_copy[22] = 0
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_h_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[18] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 1
                num_arr_copy[23] = 0
                feature_vec_dict[pre_utt_non_End+'_n_End'] = num_arr_copy

                num_arr_copy = np.copy(feature_vec_list[dialog_len-2][pre_utt])
                num_arr_copy[19] += 1
                num_arr_copy[20] = 0
                num_arr_copy[21] = 0
                num_arr_copy[22] = 0
                num_arr_copy[23] = 1
                feature_vec_dict[pre_utt_non_End+'_s_End'] = num_arr_copy

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