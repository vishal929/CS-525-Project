# the paper in question stops training using early stopping
# it stops training once avg of training and validation AUC has not increased for 15 consecutive epochs

from tensorflow import keras

class CombineTrainValAUC(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        logs['train_val_auc']=0

    def on_epoch_end(self, epoch, logs=None):
        # creating our custom monitor which is the average of train and val AUC
        logs['train_val_auc'] = (logs['auc'] + logs['val_auc'])/2