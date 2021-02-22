# Dialogical-Emotion-Decoding-CRF
* Train Weight
    * `python3 CRF_train.py [-h] [-i ITERATION] [-l LEARNING_RATE] [-d DATASET]`
    * optional arguments:
      *    -i：Set parameter update times. (default value is 200)
      *    -l：Set learning rate. (default value is 0.00001)
      *    -d：Set the dataset to be used for training. (default is Original)
            * Option 1：Original
            * Option 2：C2C (Class to class mapping by pre-trained classifier)
            * Option 3：U2U (Utt to Utt mapping by pre-trained classifier)

* After weight training, the weight value will be saved as `.pickle` in `weight` directory
* Test ACC & UAR
    * `python3 CRF_test.py`
