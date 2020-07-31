from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
import os
from sklearn.model_selection import train_test_split
from Model.Sub_block import CenterLossLayer
from Model.Sub_block import zero_loss

# load models from file
def load_all_models(n_models):
    model_name = '/media/brl/RaspberryPiData/xinwang/Model4_drop3+bn_bs256+lr3e-4+centerloss0.005+mixup0.3/'
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = model_name + 'fold' + str(i) + '.h5'
        # load model from file
        model = load_model(filename, {'CenterLossLayer': CenterLossLayer, 'zero_loss': zero_loss})
        print(model.input)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:  # 对原已训练好的模型model，冻结所有layer不再参加训练，
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    for model in members:
        print(model.input)
    ensemble_visible = []
    # for model in members:
    #     ensemble_visible.append(model[0])
    ensemble_visible = [model.input[0] for model in members]  # 获取n个原模型的input张量
    # concatenate merge output from each model
    ensemble_outputs = [model.output[0] for model in members]  # 获取n个原模型的output张量
    merge = concatenate(ensemble_outputs)  # 披了外壳的tf.concat()。参考：https://blog.csdn.net/leviopku/article/details/82380710
    hidden = Dense(64, activation='relu')(merge)
    output = Dense(19, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    #plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, input_dummpy, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	inputy_enc = to_categorical(inputy)
	# fit model
	model.fit(X, inputy_enc, epochs=5, verbose=1)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX, inputY):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)

train = pd.read_csv('sensor_train.csv')
test = pd.read_csv('sensor_test.csv')
sub = pd.read_csv('提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()
y= np.array(y)
train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5



X = np.zeros((7292, 60, 8, 1))
t = np.zeros((7500, 60, 8, 1))
t_dummy = np.zeros((7500, 19))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    X[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point'],
                                    axis=1), 60, np.array(tmp.time_point))[0]

# split into train and test
n_train = 100
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2, shuffle=True)
print(trainX.shape, testX.shape)
train_dummy = np.zeros((trainX.shape[0], 19))
test_dummy = np.zeros((testX.shape[0], 19))
# load all models
n_members = 10
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, trainX, train_dummy, trainy)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, testX, testy)
yhat = argmax(yhat, axis=1)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)