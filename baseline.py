import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample

from Model.Model1 import Net
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention

train = pd.read_csv('sensor_train.csv')
test = pd.read_csv('sensor_test.csv')
sub = pd.read_csv('提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()

train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

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

model_name = 'stored_data/Model4/'


kfold = StratifiedKFold(5, shuffle=True)

proba_t = np.zeros((7500, 19))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    y_ = to_categorical(y, num_classes=19)
    model = Net()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
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
                                   patience=20)
    checkpoint = ModelCheckpoint(model_name+f'fold{fold}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    model.fit(x[xx], y_[xx],
              epochs=200,
              batch_size=512,
              verbose=1,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[early_stopping, checkpoint])
    model.load_weights(model_name+f'fold{fold}.h5', {'SeqSelfAttention': SeqSelfAttention})
    proba_t += model.predict(t, verbose=0, batch_size=1024) / 5.

sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv(model_name+'submit1.csv', index=False)
