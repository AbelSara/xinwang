import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

from Model.Model5 import Net
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention
from Model.mixup_generator import MixupGenerator
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed()

def acc_combo(y, y_pred):
    # print(y)
    # print(y_pred)
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0

def cul_acc_combo(y, y_pred):
    sample_num = len(y)
    score_ = []
    for i in range(sample_num):
        score_.append(acc_combo(y[i],y_pred[i]))
    return np.mean(score_)

train = pd.read_csv('sensor_train.csv')
test = pd.read_csv('sensor_test.csv')
sub = pd.read_csv('提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()
y= np.array(y)
train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

model_name = 'Model5_drop3+bn_bs256+lr3e-4+mixup0.3'
if os.path.exists(model_name):
    shutil.rmtree(model_name)
os.mkdir(model_name)

x = np.zeros((7292, 60, 8, 1))
t = np.zeros((7500, 60, 8, 1))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    x[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point'],
                                    axis=1), 60, np.array(tmp.time_point))[0]

seed = 1
batch_size = 256
kfold = StratifiedKFold(10, shuffle=True, random_state=seed)
valid_pred_list = []
proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    y_ = to_categorical(y, num_classes=19)
    model = Net()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=5e-4),
                  metrics=['acc'])
    model.summary()
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=0,
                                mode='max',
                                factor=0.1,
                                patience=6)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   verbose=0,
                                   mode='max',
                                   patience=100)
    checkpoint = ModelCheckpoint(model_name+'/'+f'fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    training_generator = MixupGenerator(x[xx], y_[xx], batch_size=batch_size, alpha=0.3)()
    # for i in range(19):
    #     print(str(i) + " total in train set: "+str(sum(y[xx]==i)))
    # for i in range(19):
    #     print(str(i) + " total in valid set: " + str(sum(y[yy] == i)))
    model.fit_generator(generator=training_generator,
                    steps_per_epoch=x[xx].shape[0] // batch_size,
              epochs=500,
              #batch_size=256,
              verbose=1,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[early_stopping, checkpoint])
    model.load_weights(model_name+'/'+f'fold{fold}.h5', {'SeqSelfAttention': SeqSelfAttention})
    proba_t += model.predict(t, verbose=0, batch_size=256) / 10.
    valid_pred = model.predict(x[yy], verbose=0, batch_size=256)
    acc = cul_acc_combo(y[yy], np.argmax(valid_pred, axis=1))
    print(acc)
    valid_pred_list.append(acc)

acc_res = sum(valid_pred_list) / 10
print(acc_res)

sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv(model_name+'/'+str(acc_res)+model_name+'.csv', index=False)
