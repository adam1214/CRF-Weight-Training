# Dialogical-Emotion-Decoding-CRF
* Train Weight
    * `python3 CRF_train.py [-h] [-i ITERATION] [-l LEARNING_RATE] [-d DATASET] [-c CONCATENATION] [-n INTER_INTRA]`
    * optional arguments:
      *    -i：Set parameter update times. (default value is 3000)
      *    -l：Set learning rate. (default value is 0.000001)
      *    -d：Set the dataset to be used for training. (default is Original)
            * Option 1：Original
            * Option 2：C2C (Class to class mapping by pre-trained classifier)
            * Option 3：U2U (Utt to Utt mapping by pre-trained classifier)
      *    -c：When predicting a dialog, do you want to duplicate it 2 times and concatenate them together? 1 is yes, 0 is no. (default is 1)
      *    -n：When predicting a dialog, use intraspeaker emotion flow or interspeaker emotion change. (default is intra)

* After weight training, the weight value will be saved as `.pickle` in `weight` directory
* Test ACC & UAR
    * `python3 CRF_test.py`
