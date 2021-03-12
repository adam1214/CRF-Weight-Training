# Dialogical-Emotion-Decoding-CRF
* Train Weight
    * `python3 CRF_train.py [-h] [-i ITERATION] [-l LEARNING_RATE] [-d DATASET] [-c CONCATENATION] [-n INTER_INTRA] [-s SPEAKER_INFO_TRAIN]`
    * optional arguments:
      *    -i：Set parameter update times. (default value is 3000)
      *    -l：Set learning rate. (default value is 0.000001)
      *    -d：Set the dataset to be used for training. (default is Original)
            * Option 1：Original
            * Option 2：C2C (Class to class mapping by pre-trained classifier)
            * Option 3：U2U (Utt to Utt mapping by pre-trained classifier)
      *    -c：When predicting a dialog, do you want to duplicate it 2 times and concatenate them together? 1 is yes, 0 is no. (default is 0)
      *    -n：When predicting a dialog, use intraspeaker emotion flow or interspeaker emotion change. (default is intra)
      *    -s：When estimating emotion transition probabilities, do you want to consider speakers information? (default is 2)
            * 0：not consider speaker info
            * 1：consider intra-speaker only
            * 2：consider intra-speaker & inter-speaker
      *    -f：Set training batch size. Setting -1 means that flexible training batch size. Note that the fixed-size dialog segment is all the same speaker ID. (default is 15)

* After weight training, the weight value will be saved as `.pickle` in `weight` directory
* Test ACC & UAR
    * `python3 CRF_test.py`
