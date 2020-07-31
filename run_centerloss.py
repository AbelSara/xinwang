import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
import keras
import keras.backend as K
from Model.Model4 import Net
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.callbacks import TensorBoard
from keras_self_attention import SeqSelfAttention
from Model.Sub_block import zero_loss
from Model.mixup_generator import MixupGenerator_center
from sklearn.metrics import confusion_matrix as confusion
from Model.random_eraser import get_random_eraser
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
import time

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

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

def cul_acc_combo(y, y_pred):
    sample_num = len(y)
    score_ = []
    for i in range(sample_num):
        score_.append(acc_combo(y[i],y_pred[i]))
    return np.mean(score_)

def jitter(x, snr_db):
    """
    根据信噪比添加噪声
    :param x:
    :param snr_db:
    :return:
    """
    # 随机选择信噪比
    assert isinstance(snr_db, list)
    snr_db_low = snr_db[0]
    snr_db_up = snr_db[1]
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]

    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x ** 2, axis=0, keepdims=True) / x.shape[0]  # 计算信号功率
    Np = Xp / snr  # 计算噪声功率
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)  # 计算噪声
    xn = x + n
    return xn

def standardization(X, mean, std):
    x1 = X.transpose(0, 1, 3, 2)
    #x1 = X
    x2 = x1.reshape(-1, x1.shape[-1])
    # mean = [8.03889039e-03, -6.41381949e-02, 2.37856977e-02, 8.64949391e-01,
    #         2.80964889e+00, 7.83041714e+00, 6.44853358e-01, 9.78580749e+00]
    # std = [0.6120893, 0.53693888, 0.7116134, 3.22046385, 3.01195336, 2.61300056, 0.87194132, 0.68427254]
    mu=np.array(mean)
    sigma=np.array(std)
    x3 = ((x2 - mu) / (sigma))
    x4 = x3.reshape(x1.shape).transpose(0, 1, 3, 2)
    #x4 = x3.reshape(x1.shape)
    return x4
def cul_mean_std(X):
    X = X.reshape((-1, 8))
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s

def res_vote(pred_list):
    res = []
    sample_len = pred_list[0].shape[0]
    for i in range(sample_len):
        count = np.zeros(19)
        for pred in pred_list:
            count[pred[i]] += 1
        res.append(count.argmax())
    return res


train = pd.read_csv('sensor_train.csv')
test = pd.read_csv('sensor_test.csv')
sub = pd.read_csv('提交结果示例.csv')
y = train.groupby('fragment_id')['behavior_id'].min()
y= np.array(y)
train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

model_name = '/media/brl/RaspberryPiData/xinwang/Model4_drop3+bn_bs256+lr3e-4+centerloss0.005+mixup0.3+3seed/'
if not os.path.exists(model_name):
    os.mkdir(model_name)

x = np.zeros((7292, 60, 8, 1))
t = np.zeros((7500, 60, 8, 1))
t_dummy = np.zeros((7500, 19))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    x[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
# mean1, std1 = cul_mean_std(x)
# mean2, std2 = cul_mean_std(t)
# x = x.reshape((7292*60, 8))
# t = t.reshape((7500*60, 8))
# print('mean of train set: '+str(np.mean(x, axis=0)))
# print('mean of test set: '+str(np.mean(t, axis=0)))
# print('std of train set: '+str(np.std(x, axis=0)))
#
# print('std of test set: '+str(np.std(t, axis=0)))
# x = x.reshape((7292, 60, 8, 1))
# t = t.reshape((7500, 60, 8, 1))
# x = standardization(x, mean1, std1)
# t = standardization(t, mean2, std2)
# #x = jitter(x, [-1,1])
# x = x.reshape((7292*60, 8))
# t = t.reshape((7500*60, 8))
# print('after norm mean of train set: '+str(np.mean(x, axis=0)))
# print('after norm mean of test set: '+str(np.mean(t, axis=0)))
# print('after norm std of train set: '+str(np.std(x, axis=0)))
#
# print('after norm std of test set: '+str(np.std(t, axis=0)))
# x = x.reshape((7292, 60, 8, 1))
# t = t.reshape((7500, 60, 8, 1))
# scaler = MinMaxScaler((-1, 1))
# x = x.reshape((7292*60, 8))
# t = t.reshape((7500*60, 8))
# dataset = np.concatenate([x, t], axis=0)
# print('mean of train set: '+str(np.mean(x, axis=0)))
# print('mean of test set: '+str(np.mean(t, axis=0)))
# print('std of train set: '+str(np.std(x, axis=0)))
# print(x[0, :, 0])

# print('std of test set: '+str(np.std(t, axis=0)))
# print('mean of whole set: '+str(np.mean(dataset, axis=0)))
# print('std of whole set: '+str(np.std(dataset, axis=0)))
# print(t[0, :, 0])
# scaler.fit(x)
# x = scaler.transform(x)
# t = scaler.transform(t)
# x = x.reshape((7292, 60, 8, 1))
# t = t.reshape((7500, 60, 8, 1))


# print(x[0, :, 0, 0])
# print(t[0, :, 0, 0])
lambda_centerloss=0.005
# seed = 1
batch_size = 256

valid_pred_list = []
proba_t = np.zeros((7500, 19))
warmup_epoch = 10
learning_rate_base = 0.001
epochs = 1000
noise_SNR_db = [-5, 15]
pred_list = []
# t = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
count = 0
seed = 1
# for seed in seeds:
kfold = StratifiedKFold(10, shuffle=True, random_state=seed)
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    # mu, sigma = cul_mean_std(x[xx])
    # print(mu)
    # print(sigma)
    x_train = x[xx]
    x_val = x[yy]
    # x_train = standardization(x[xx], mu, sigma)
    # x_train = jitter(x_train, noise_SNR_db)
    # x_val = standardization(x[yy], mu, sigma)
    # t_standard = standardization(t, mu, sigma)

    y_ = to_categorical(y, num_classes=19)
    y_train = y_[xx]
    y_val = y_[yy]
    dummy1 = np.zeros((x_train.shape[0], 1))
    dummy2 = np.zeros((x_val.shape[0], 1))
    sample_count = x_train.shape[0]
    total_steps = int(epochs * sample_count / batch_size)
    # Compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * sample_count / batch_size)
    model = Net()
    model.compile(loss=['categorical_crossentropy', zero_loss],
                  loss_weights=[1, lambda_centerloss],

                  optimizer=Adam(learning_rate=3e-4),
                  metrics=['acc'])
    model.summary()
    # Compute the number of warmup batches.
    # warmup_batches = warmup_epoch * sample_count / batch_size

    # Create the Learning rate scheduler.
    # warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
    #                                         total_steps=total_steps,
    #                                         warmup_learning_rate=0.0,
    #                                         warmup_steps=warmup_steps,
    #                                         hold_base_rate_steps=0)
    plateau = ReduceLROnPlateau(monitor="val_acc",
                                verbose=1,
                                mode='max',
                                factor=0.5,
                                patience=20,
                                min_lr=1e-4)
    early_stopping = EarlyStopping(monitor='val_behaviour_acc',
                                   verbose=0,
                                   mode='max',
                                   patience=200)
    checkpoint = ModelCheckpoint(model_name+f'model{count}.h5',
                                 monitor='val_behaviour_acc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    # tensorboard = TensorBoard(
    #     log_dir='./Graph/' + t,
    #     histogram_freq=0,
    #     write_graph=True,
    #     write_images=True
    # )
    # datagen = ImageDataGenerator(
    #     height_shift_range=0.1
    #     # preprocessing_function=get_random_eraser(v_l=0, v_h=1)
    # )
    training_generator = MixupGenerator_center(x_train, y_train, dummy1, batch_size=batch_size, alpha=0.3)()
    # for i in range(19):
    #     print(str(i) + " total in train set: "+str(sum(y[xx]==i)))
    # for i in range(19):
    #     print(str(i) + " total in valid set: " + str(sum(y[yy] == i)))
    # model.fit_generator(generator=training_generator,
    #                 steps_per_epoch=x_train.shape[0] // batch_size,
    #           epochs=1000,
    #           # batch_size=256,
    #           verbose=1,
    #           shuffle=True,
    #           validation_data=([x_val, y_val], [y_val, dummy2]),
    #           callbacks=[checkpoint])
    model.load_weights(model_name+f'model{count}.h5', {'SeqSelfAttention': SeqSelfAttention})
    p, _ = model.predict([t, t_dummy], verbose=0, batch_size=batch_size)
    # sub.behavior_id = np.argmax(p, axis=1)
    # pred_list.append(np.argmax(p, axis=1))
    # sub.to_csv(model_name+'fold'+str(fold)+'.csv', index=False)
    proba_t += p / 30.
    valid_pred, _ = model.predict([x_val, y_val], verbose=0, batch_size=batch_size)
    valid_pred = np.argmax(valid_pred, axis=1)
    acc = cul_acc_combo(y[yy], valid_pred)
    print(acc)
    valid_conf = confusion(valid_pred, y[yy])
    print(valid_conf)
    valid_pred_list.append(acc)
    count += 1

print(valid_pred_list)
acc_res = sum(valid_pred_list) / 30
print(acc_res)
# vote_res = res_vote(pred_list)
np.save(model_name+'proba.npy', proba_t)
sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv(model_name+str(acc_res)+'_drop3_bn_bs256+lr3e-4+centerloss0.005+mixup0.3+3seed.csv', index=False)

# sub.behavior_id = vote_res
# sub.to_csv(model_name+str(acc_res)+'_drop3_bn_bs256+lr3e-4+centerloss0.005+mixup0.3+vote.csv', index=False)
