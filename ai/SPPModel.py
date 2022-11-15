import sys
import logging


logger = logging.getLogger(__name__)
is_64bits = sys.maxsize > 2**32
if not is_64bits:
    logger.critical('64bit 환경으로 실행해 주시기 바랍니다.')
    exit(1)

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer, MaxAbsScaler,RobustScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout , Activation, GRU

from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow as tf
#https://github.com/linewalks/AngularGrad-tf 참고 
from .angular_grad import AngularGrad
from matplotlib import pyplot
####
from keras.layers import GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation, Add, Flatten, BatchNormalization , GlobalAveragePooling1D
from keras.layers import Dense, Embedding, Input, concatenate, TimeDistributed, Attention, PReLU, ELU
#### 2022 11 01 add
## pip install dropconnect-tensorflow (Successfully installed dropconnect-tensorflow-0.1.1)
from dropconnect_tensorflow import DropConnectDense
from .attention.DropConnect import DropConnect
from tcn import TCN
from keras.regularizers import l2


from .attention_3d_block.attention_3d_block import attention_3d_block2

#### 2022-11-02
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from pyMetaheuristic.algorithm import cuckoo_search
from pyMetaheuristic.utils import graphs
from tensorflow.keras.losses import Huber

from tensorflow.keras.utils import plot_model

import keras
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#from .plotlibrary.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pyd

from .Transformer.Transformer import *

import time
import os

# 20221114 add
from keras.initializers import glorot_uniform,GlorotUniform
from wandb.keras import WandbCallback
import wandb


plt.rcParams['font.family'] = 'Malgun Gothic'


class DataNotEnough(BaseException):
    pass

# TensorBoard 결과 로그 폴더 생성
if not os.path.isdir("logs"):
    os.mkdir("logs")


# 참고 - https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb#scrollTo=4g2E9J4GndjH
# 참고 -https://rinha7.github.io/keras-callbacks/
def lr_scheduler(epoch, lr):
    # log the current learning rate onto W&B
    if wandb.run is None:
        raise wandb.Error("You must call wandb.init() before WandbCallback()")

    wandb.log({'learning_rate': lr}, commit=False)
    
    if epoch < 7:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



# 학습 함수
def train(data, model, n_epochs=100, batch_size=70, verbose=1):
    #1106 Add
    date_now = time.strftime("%Y-%m-%d")
    model_function_name= "attention_model_1114_v3"

    model_name = f"{date_now}_{model_function_name}"
    checkpoint_filepath = 'ModelCheckpoint/attention_model_1114_v3/Checkpoint'

    early_stopping = EarlyStopping(monitor='val_loss', patience=1000)  # patience 번이상 더 좋은 결과가 없으면 학습을 멈춤
    #callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
    #                                          monitor='val_loss', 
    #                                          save_best_only=True, verbose=1)

    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, verbose=1, mode='min',monitor='val_loss')
    # val_loss 인 경우, loss 값이기 때문에 값이 작을수록 좋습니다. 따라서 이때는 min을 입력해줘야합니다.
    # monitor=>모델을 저장할 때, 기준이 되는 값을 지정합니다.
    # 예를 들어, validation set의 loss가 가장 작을 때 저장하고 싶으면 'val_loss'를 입력하고
    # 만약 train set의 loss가 가장 작을 때 모델을 저장하고 싶으면 'loss'를 입력합니다


    #tensorboard --logdir="logs" 
    # verbose 옵션은 실행 과정을 콘솔에 띄워줄지 말지에 대한 옵션
    # 0 - 끔, 1 - 움직이는 실시간 그래프, 2 - 정적 메시지
    # history = model.fit(data["X_train"], data["y_train"],
    #                     batch_size=batch_size,
    #                     epochs=n_epochs,
    #                     validation_data=(data["X_test"], data["y_test"]),
    #                     callbacks=[early_stopping],
    #                     verbose=verbose)
    
# api key :483c55b5c6488e6484b5173b3f6dfe92af598e2d
# wandb:  View project at https://wandb.ai/aiinvestmentbot/test-project
# wandb:  View run at https://wandb.ai/aiinvestmentbot/test-project/runs/1mwzy32e
    wandb.init(project="test-project", entity="aiinvestmentbot")
    wandb_callback = WandbCallback(monitor='val_loss',save_model=True,mode='min',log_weights=True,log_evaluation=True,validation_steps=5,verbose=1)
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.0001, mode='min',verbose=1)
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                    patience=1, verbose=1, mode='min',
#                                    min_delta=0.0001, cooldown=0, min_lr=1e-8)

    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[tensorboard,early_stopping,ModelCheckpoint,wandb_callback,reduce_lr],
                        verbose=verbose)
    
    return history

# 에러 평가 함수 # 스케일링된 결과 값을 본래 값으로 복원한다 (inverse_transform 함수란?)
# 참고 사이트
#https://stackoverflow.com/questions/48775305/what-function-defines-accuracy-in-keras-when-the-loss-is-mean-squared-error-mse

# https://www.kaggle.com/code/ajax0564/transfromer-timetovector-timeseries
def evaluate(data, model):
    # 가중치 로드
    model.load_weights("ModelCheckpoint/attention_model_1114_v3/Checkpoint")
    

    train_huber_loss, train_mae, train_rmse  =  model.evaluate(data["X_train"], data["y_train"], verbose=1)
    test_huber_loss, test_mae, test_rmse  = model.evaluate(data["X_test"], data["y_test"], verbose=1)
    
    # train_mae = data["column_scaler"]["close"].inverse_transform([[train_mae]])[0][0]
    # train_rmse = data["column_scaler"]["close"].inverse_transform([[train_rmse]])[0][0]
   

    # test_mae = data["column_scaler"]["close"].inverse_transform([[test_mae]])[0][0]
    # test_rmse = data["column_scaler"]["close"].inverse_transform([[test_rmse]])[0][0]
    train_mae = data["column_scaler"]["close"].inverse_transform([[train_mae]])[0][0]
    train_rmse = data["column_scaler"]["close"].inverse_transform([[train_rmse]])[0][0]
   

    test_mae = data["column_scaler"]["close"].inverse_transform([[test_mae]])[0][0]
    test_rmse = data["column_scaler"]["close"].inverse_transform([[test_rmse]])[0][0]
    #np.mean


    return train_huber_loss, train_mae, train_rmse,test_huber_loss, test_mae, test_rmse




# def evaluate(data, model):
#     results = pd.DataFrame({'r2_score':r2_score(data["X_test"], data["y_test"]),}, index=[0])
#     results['mean_absolute_error'] = mean_absolute_error(data["X_test"], data["y_test"])
#     results['median_absolute_error'] = median_absolute_error(data["X_test"], data["y_test"])
#     results['mse'] = mean_squared_error(data["X_test"], data["y_test"])
#     results['msle'] = mean_squared_log_error(data["X_test"], data["y_test"])
#     results['mape'] = mean_absolute_percentage_error(data["X_test"], data["y_test"])
#     results['rmse'] = np.sqrt(results['mse'])
#     return results
    
# def mean_absolute_percentage_error(y_true, y_pred): 
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 예측 주가를 계산 해주는 함수
def predict(data, model, n_steps=1):
    last_sequence = data["last_sequence"][-n_steps:]
    column_scaler = data["column_scaler"]
    # last_sequence를 reshape 합니다   
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # 3차원으로 변경
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # 스케일 된 예측값을 계산 (0과 1사이의 값)
    prediction = model.predict(last_sequence)
    # 스케일 된 값에서 실제 값으로 변환
    predicted_price = column_scaler["close"].inverse_transform(prediction)[0][0]
    
    return predicted_price
    

# 스케일링 and "X_train", "X_test", "y_train", "y_test" 추출 함수
def load_data(df, n_steps=1, lookup_step=1, test_size=0.3, shuffle=True):
    # return 해줘야 할 모든 값들은 result 변수에 넣을 예정
    result = {}

    column_scaler = {}
    # data를 칼럼별로 0과 1사이의 값으로 scale
    for column in df.columns:  # close, volume, open, high, low 컬럼들을 모두 MinMaxScaler 해준다.
        scaler = MinMaxScaler()
        #scaler = QuantileTransformer()
        
        #scaler = tf.keras.utils.normalize(column_scaler, axis=0)
        df[column] = scaler.fit_transform(np.expand_dims(df[column].to_numpy(), axis=1))
        
        # 해당 칼럼에 쓰인 scaler를 저장한다
        column_scaler[column] = scaler

    # result에 칼럼별 scaler들을 넣어준다
    result["column_scaler"] = column_scaler
    last_sequence = np.array(df.tail(lookup_step))
    df['future'] = df['close'].shift(-lookup_step)
    # drop NaNs (비어있는 row는 제거)
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df.loc[:, df.columns != 'future'].to_numpy(), df['future'].to_numpy()):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    if not sequence_data:
        raise DataNotEnough()

    # dataset의 뒤에서 lookup_step 만큼 짤라온 last_sequence와 기존의 sequence를 합쳐서 last_sequence를 만듦
    # last_sequence는 향후 dataset에 없는 미래데이터 예측에 쓰임
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # 결과 값을 result에 넣어준다
    result['last_sequence'] = last_sequence

    X, y = [], []

    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # numpy로 변환
    X = np.array(X)
    y = np.array(y)
    # 신경망에 적용(fit)하기위해 X의 shape을 변경
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # dataset을 train용도와 test 용도에 맞춰 나누어 줍니다.
    # test_size 가 0.2 이면 X_train, y_train에 80% 데이터로 트레이닝 하고 X_test,y_test에 나머지 20%로 테스트를 하겠다는 의미
    # result["X_train"] , result["X_test"]  변수는 3차원이고 result["y_train"], result["y_test"] 1차원이다
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = \
        train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    return result

# 모델 생성 함수
def create_model(maxlen= 21,units=50, dropout=0.3, n_steps=5, LOSS = "mae", optimizer='cos', n_layers=4, cell=LSTM):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(cell(units, return_sequences=True, input_shape=(maxlen, n_steps)))
        elif i == n_layers - 1:  # 마지막 layer
            model.add(cell(units))
        else:
            model.add(cell(units, return_sequences=True))
        # 매 layer마다 dropout을 해줌
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    model.summary()

    return model

def create_model_LSTM(maxlen=21,units=50, dropout=0.3, n_steps=1, LOSS = "mae", optimizer='cos', n_layers=4, cell=LSTM):
    model=Sequential()
    # model.add(Conv1D(64,kernel_size=3,activation='relu',input_shape=(100,1)))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(50,kernel_size=5,activation='relu'))
    model.add(LSTM(units,return_sequences=True,input_shape=(maxlen, n_steps)))
    model.add(tf.keras.layers.Bidirectional(LSTM(units,return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(LSTM(units,return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(LSTM(units)))
    # model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    # # model.add(LSTM(50,return_sequences=True))
    # model.add(LSTM(50,return_sequences=True))
    # model.add(LSTM(50))
    # model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    model.summary()

    return model

#A bidirectional long short-term memory (BiLSTM) network 
# is a combination of two LSTMs, i.e., forward and backward.
# Based on LSTM, BiLSTM can extract the feature of forward and backward simultaneously
# 2022-10-25 Written by SEONGJAE-YOO (Commits on Oct 25, 2022)
def create_model_Bidirectional(maxlen=21,dropout=0.5, n_steps=1, LOSS = "mae", optimizer= 'cos', n_layers=4, cell=LSTM):
    
    model = Sequential()
    for i in range(n_layers):
            if i ==0:
                model.add(tf.keras.layers.Bidirectional(cell(128,return_sequences=True, activation="tanh") , input_shape=(maxlen, n_steps)))          
            elif i == n_layers - 1:  # 마지막 layer
                model.add(tf.keras.layers.Bidirectional(cell(64, return_sequences=False,activation="tanh")))
            elif i == n_layers - 2:  # 마지막 2번째 layer
                model.add(tf.keras.layers.Bidirectional(cell(128,return_sequences=True, activation="tanh")))    
            else:
                model.add(tf.keras.layers.Bidirectional(cell(256,return_sequences=True,activation="tanh")))
            # 매 layer마다 dropout을 해줌
            model.add(Dropout(dropout))
    model.add(tf.keras.layers.Dense(1)) # tf.keras.layers.Activation(tf.nn.relu)
    #model.add(activation ='softmax')  # model.add(layers.Dense(64, activation='relu'))
    #model.compile(loss = tf.keras.losses.CategoricalCrossentropy() , optimizer= tf.keras.optimizers.RMSprop(learning_rate=1e-3))
    
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    model.summary()

   
      
#To prevent overfitting,
#the dropout technique was used. The dropout technology stops the hidden layer neurons with
#self-defined probability numbers from working in the forward propagation of the training process 
  
        # # With custom backward layer
        # model = Sequential()
        # forward_layer = LSTM(10, return_sequences=True)
        # backward_layer = LSTM(10, activation='relu', return_sequences=True,
        #                     go_backwards=True)
        # model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
        #                         input_shape=(5, 10)))
        # model.add(Dense(5))
        # model.add(Activation('softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


    return model    

# ####
# def create_model_Bidirectional_2(units=32, dropout=0.3, n_steps=20, loss = 'mse', optimizer= 'RMSprop', n_layers=4, cell=LSTM):
#     maxlen = None
#     embed_size =100 
#     recurrent_units = 64
#     recurrent_dropout_rate = 0.5 # dropout 비율과 같이 설정 
#     dense_size =  32
    
#     model = Sequential()
    
           
#     x , input_layer = tf.keras.layers.Bidirectional(GRU(128,return_sequences=True, activation="relu") , input_shape=(None, n_steps))          
#     x = Dropout(dropout)(x)
#     x = tf.keras.layers.Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout,
#                            recurrent_dropout=recurrent_dropout_rate))(x)
#     #x = AttentionWeightedAverage(maxlen)(x)
#     x_a = GlobalMaxPool1D()(x)
#     x_b = GlobalAveragePooling1D()(x)
#     #x_c = AttentionWeightedAverage()(x)
#     #x_a = MaxPooling1D(pool_size=2)(x)
#     #x_b = AveragePooling1D(pool_size=2)(x)
#     x = concatenate([x_a,x_b], axis=1)
#     #x = Dense(dense_size, activation="relu")(x)
#     #x = Dropout(dropout_rate)(x)
#     x = Dense(dense_size, activation="relu")(x)
#     output_layer = Dense(5, activation="sigmoid")(x)

#     model = Model(inputs=input_layer, outputs=output_layer)            
    
#     model.compile(loss=loss, metrics=[tf.keras.metrics.MeanSquaredError(), 'accuracy'], optimizer=AngularGrad(optimizer))
#     model.summary()


#     return model   


####

# def create_model_GRU(units=128, dropout=0.3, n_steps=20, loss = 'mse', optimizer= 'RMSprop', n_layers=4, cell=GRU):
#     maxlen = None
#     embed_size =100 
#     recurrent_units = 64
#     recurrent_dropout_rate = 0.5 # dropout 비율과 같이 설정 
#     dense_size =  32
#     #input_layer = Input(shape=(maxlen,))
#     input_layer = Input(shape=(maxlen, embed_size), )
#     #embedding_layer = Embedding(max_features, embed_size, 
#     #                            weights=[embedding_matrix], trainable=False)(input_layer)
#     x = tf.keras.layers.Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout,
#                            recurrent_dropout=recurrent_dropout_rate))(input_layer)
#     x = Dropout(dropout)(x)
#     x = tf.keras.layers.Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout,
#                            recurrent_dropout=recurrent_dropout_rate))(x)
#     #x = AttentionWeightedAverage(maxlen)(x)
#     x_a = GlobalMaxPool1D()(x)
#     x_b = GlobalAveragePooling1D()(x)
#     #x_c = AttentionWeightedAverage()(x)
#     #x_a = MaxPooling1D(pool_size=2)(x)
#     #x_b = AveragePooling1D(pool_size=2)(x)
#     x = concatenate([x_a,x_b], axis=1)
#     #x = Dense(dense_size, activation="relu")(x)
#     #x = Dropout(dropout_rate)(x)
#     x = Dense(dense_size, activation="relu")(x)
#     output_layer = Dense(5, activation="sigmoid")(x)

#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(loss=loss, metrics=[loss], optimizer=AngularGrad(optimizer))
#   #  model.compile(loss=loss, metrics=[tf.keras.metrics.MeanSquaredError(), 'accuracy'], optimizer=AngularGrad(optimizer))
#     model.summary()


####
# LSTM + conv
# 2022-10-25 Written by SEONGJAE-YOO (Commits on Oct 25, 2022)
def create_lstm_cnn(maxlen=21, units=32, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen,n_steps))
    #input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix],
    #              trainable=False)(inp)
    x = LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(input_layer)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=units, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=300,
                       kernel_size=5,
                       padding='valid',
                       activation='relu',
                       strides=1)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # x = Conv1D(filters=300,
    #                   kernel_size=5,
    #                   padding='valid',
    #                   activation='relu',
    #                   strides=1)(x)
    # x = MaxPooling1D(pool_size=2)(x)

    # x = Conv1D(filters=300,
    #                   kernel_size=3,
    #                   padding='valid',
    #                   activation='tanh',
    #                   strides=1)(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a,x_b])

    x = Dense(16, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=input_layer, outputs=x, name='lstm_cnn')
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model


####

# 2022-10-26 Written by SEONGJAE-YOO (Commits on Oct 26, 2022)
# LSTM-CNN-version2
def create_dpcnn(maxlen=21, units=524, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen,n_steps))
   
    X_shortcut1 = LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(input_layer)
    X_shortcut1 = Dropout(dropout)(X_shortcut1)
    # first block
    X_shortcut1 = Conv1D(filters=units, kernel_size=1, padding='same',strides=3)(X_shortcut1)
    X_shortcut1 = Activation('relu')(X_shortcut1)
    X_shortcut1 = Conv1D(filters=units, kernel_size=1, padding='same',strides=3)(X_shortcut1)
    X_shortcut1 = Activation('relu')(X_shortcut1)


    # # connect shortcut to the main path
    # X = Activation('relu')(input_layer)  # pre activation
    # X = Add()([X,X_shortcut1])
    X_shortcut1 = MaxPooling1D(pool_size=1, strides=2, padding='same')(X_shortcut1)


    # second block
    X_shortcut2 = X_shortcut1
    X_shortcut2 = Conv1D(filters=units, kernel_size=1, strides=3)(X_shortcut2)
    X_shortcut2 = Activation('relu')(X_shortcut2)
    X_shortcut2 = Conv1D(filters=units, kernel_size=1, strides=3)(X_shortcut2)
    X_shortcut2 = Activation('relu')(X_shortcut2)

    # connect shortcut to the main path
    X_shortcut2 = MaxPooling1D(pool_size=1, strides=2, padding='same')(X_shortcut2)

    # Output
    #X = Flatten()(X)

    x_a = GlobalMaxPool1D()(X_shortcut2)
    x_b = GlobalAveragePooling1D()(X_shortcut2)
    X  = concatenate([x_a,x_b])

    X = Dense(1)(X)

    model = Model(inputs = input_layer, outputs = X, name='dpcnn')
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    return model

# 2022-10-26 Written by SEONGJAE-YOO (Commits on Oct 26, 2022)
#####GRU-CNN
## cnn3 
def create_GRU_CNN(maxlen=21, units=32, dropout=0.3, n_steps=5, LOSS = "mae", optimizer= 'cos'):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, n_steps), )
    x = GRU(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(input_layer)
    #x = Dropout(dropout_rate)(x) 

    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    #x = Dropout(dropout_rate)(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=input_layer, outputs=x,name='GRU-CNN')
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model

######
# 2022-10-26 Written by SEONGJAE-YOO (Commits on Oct 26, 2022)
# CNN_GRU 
def create_cnn_GRU(maxlen=21, units=32, dropout=0.3, n_steps=5, LOSS = "mae", optimizer= 'cos'):

    input_layer = Input(shape=(maxlen, n_steps), )


    x = Dropout(dropout)(input_layer) 
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = GRU(units)(x)
    x = Dropout(dropout)(x)
    #x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=input_layer, outputs=x, name ='CNN_GRU')
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
                
    return model

####
# Bidirectional GRU + Bidirectional LSTM
def Create_Bidirectional_GRU_LSTM(maxlen=21, units=32, dropout=0.3, n_steps=5, LOSS = "mae", optimizer= 'cos'):
    
    input_layer = Input(shape=(maxlen, n_steps), )
   
    x = tf.keras.layers.Bidirectional(GRU(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout))(input_layer)
    x = Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout))(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a,x_b])

    #x = Dense(32, activation="relu")(x)
    output_layer = Dense(1, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    return model

#### 2022-10-28 Written by SEONGJAE-YOO (Commits on Oct 28, 2022)
# Bidirectional LSTM + GRU + LSTM_ cnn
def Create_BidirectionalLSTM_GRU_LSTM(maxlen=21, units=32, dropout=0.3, n_steps=5, LOSS = "mae", optimizer= 'cos'):
    
    input_layer = Input(shape=(maxlen, n_steps), )
   
    x = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout))(input_layer)
    x = Dropout(dropout)(x)

    x = GRU(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(x)

    x = Dropout(dropout)(x)

    x = LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(x)  

    x = Dropout(dropout)(x)      

    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    #x = Dropout(dropout_rate)(x)
   # x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x) 
    model = Model(inputs=input_layer, outputs=x,name='BidirectionalLSTM_GRU_LSTM_CNN')
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model    


# # 0.010477105(mae)
# def create_filter_kernels_conv(maxlen=21, units=32, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    
#     input_layer = Input(shape=(maxlen, n_steps), )

#     conv = Conv1D(filters = units, kernel_size = 1, activation='relu')(input_layer)
#     conv = MaxPooling1D(pool_size=1)(conv)
#     conv1 = Conv1D(filters = units, kernel_size = 1, activation='relu')(conv)
#     conv1 = MaxPooling1D(pool_size=1)(conv1)
#     conv2 = Conv1D(filters = units, kernel_size = 1, activation='relu')(conv1)
#     conv3 = Conv1D(filters = units, kernel_size = 1, activation='relu')(conv2)
#     conv4 = Conv1D(filters = units, kernel_size = 1, activation='relu')(conv3)
#     conv5 = Conv1D(filters = units, kernel_size = 1, activation='relu')(conv4)
#     conv5 = MaxPooling1D(pool_size=1)(conv5)
#     conv5 = Flatten()(conv5)
#     z = Dropout(dropout)(Dense(units)(conv5))
#     #x = GlobalMaxPool1D()(x)
#     x = Dense(1)(z)
#     model = Model(inputs=input_layer, outputs=x)
#     model.summary()  
#     model.compile(loss=LOSS, 
#                 optimizer=AngularGrad(optimizer),  
#                 metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
#     return model

# 0.01242176(mae)
# Add
# # MaxPooling1D strides=
# x_a = GlobalMaxPool1D()(conv5)
#x_b = GlobalAveragePooling1D()(conv5)
#conv5 = concatenate([x_a,x_b])
def create_filter_kernels_conv(maxlen=21, units=21, dropout=0, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    
    input_layer = Input(shape=(maxlen, n_steps), )

    conv = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30, kernel_initializer="he_uniform")(input_layer)
    conv = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(conv) 
    conv = MaxPooling1D(pool_size=1,strides=2)(conv)
    conv1 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30,kernel_initializer="he_uniform")(conv)
    conv1 = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(conv1) 
    conv1 = MaxPooling1D(pool_size=1,strides=2)(conv1)
    # conv2 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv1)
    # conv3 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv2)
    # conv4 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv3)
    conv5 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30,kernel_initializer="he_uniform")(conv1)
    conv5 = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(conv5) 
    #conv5 = MaxPooling1D(pool_size=1,strides=2)(conv5)
    x_a = GlobalMaxPool1D()(conv5)
    x_b = GlobalAveragePooling1D()(conv5)
    conv5 = concatenate([x_a,x_b])
   # conv5 = Flatten()(conv5)
    z = Dropout(dropout)(Dense(units)(conv5))
    #x = GlobalMaxPool1D()(x)
    x = Dense(1)(z)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    return model

#### CNN-BiLSTM-Attention model


def create_filter_kernels_conv_v2(maxlen=21, units=32, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    #filter_kernels = [7, 7, 5, 5, 3, 3]
    input_layer = Input(shape=(maxlen, n_steps), )

    conv = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(input_layer)
    conv = MaxPooling1D(pool_size=1)(conv)
    conv1 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv)
    conv1 = MaxPooling1D(pool_size=1)(conv1)
    conv2 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv1)
    conv3 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv2)
    conv4 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv3)
    conv5 = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu)(conv4)
    conv5 = MaxPooling1D(pool_size=1)(conv5)

    z = Dropout(dropout)(Dense(units)(conv5))
    #x = GlobalMaxPool1D()(x)
    lstm_out = Dropout(0.3)(z)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    x = Dense(1)(attention_mul)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    return model



"""
ELU의 특징은 다음과 같음
ReLU의 장점을 모두 포함
Dying ReLU 문제 해결
출력값이 거의 zero-centered함
ReLU, Leaky ReLU와 달리 exp()에 대한 미분값을 계산해야 하는 비용이 발생
"""


# cnn-bilstm-attention
def attention_model(maxlen=21, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )

    x = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30, kernel_initializer="he_uniform")(input_layer)
    #x = Dropout(dropout)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    x = MaxPooling1D(pool_size=1,strides=2)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    
    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)),name='bilstm')(x)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='linear')(attention_mul) # linear 성능 향상에 꼭 필요함
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    visualize_model(model)
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model

# bilstm + attention + cnn / # 481	0.003097124 
def attention_model_1114(maxlen=21, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )

    
    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)),name='bilstm')(input_layer)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)

    x = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30, kernel_initializer="he_uniform")(attention_mul)
    #x = Dropout(dropout)(x)
    x = MaxPooling1D(pool_size=1,strides=2)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    x = Flatten()(x)

    output = Dense(1, activation='linear')(x) # linear 성능 향상에 꼭 필요함
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    visualize_model(model)
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model


# cnn-attention-bilstm / 351	0.002917611 
def attention_model_1114_v2(maxlen=21, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )

    x = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30, kernel_initializer="he_uniform")(input_layer)
    #x = Dropout(dropout)(x)
    x = MaxPooling1D(pool_size=1,strides=2)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    
   
    attention_mul = attention_3d_block2(x)
    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)),name='bilstm')(attention_mul)
    lstm_out = Dropout(dropout)(lstm_out)
    
    x = Flatten()(lstm_out) # Flatten이 (concatenate([x_a,x_b]))보다 더 좋음

    output = Dense(1, activation='linear')(x) # linear 성능 향상에 꼭 필요함
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    visualize_model(model)
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model


# cnn-attention-bilstm / (best model)


# kernel_initializer=glorot_uniform(seed=0) 대한 설명 (참고 - https://github.com/keras-team/keras-docs-ko/blob/master/sources/initializers.md)
# Glorot 균등분포 방식으로 파라미터의 초기값을 생성합니다. Xavier 균등분포 방식이라고도 불리며, 가중치 텐서의 크기에 따라 값을 조절하는 방식의 하나입니다.

# [-limit, limit]의 범위를 가진 균등분포로부터 값이 선택됩니다. 가중치 텐서의 입력 차원 크기를 fan_in, 출력 차원 크기를 fan_out이라고 할 때, limit은 sqrt(6 / (fan_in + fan_out))으로 구합니다.
def attention_model_1114_v3(maxlen=21, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )

    x = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30, kernel_initializer=glorot_uniform(seed=0))(input_layer)
    #x = Dropout(dropout)(x)
    x = MaxPooling1D(pool_size=1,strides=2)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    
    # bilstm은 kernel_initializer=glorot_uniform(seed=0) 적용하면 성능이 낮아짐
    attention_mul = attention_3d_block2(x)
    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001),kernel_initializer=GlorotUniform(seed=1)),name='bilstm')(attention_mul)
    lstm_out = Dropout(dropout)(lstm_out)
    x = Flatten()(lstm_out)
    # x_a = GlobalMaxPool1D()(lstm_out) 
    # x_b = GlobalAveragePooling1D()(lstm_out)
    # x = concatenate([x_a,x_b])

    output = Dense(1, activation='linear')(x) # linear 성능 향상에 꼭 필요함
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    visualize_model(model)
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model

#  kernel_initializer="he_uniform" 없는게 더 성능 좋음
def attention_model_v2(maxlen=21, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )
    #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01) 참고 사이트
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-keras.md
    x = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30,kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),bias_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),activity_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001))(input_layer)
    #x = Dropout(dropout)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=1e-06)(x)  # BatchNormalization 대신 LayerNormalization 사용하면?
    x = MaxPooling1D(pool_size=1,strides=2)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    
    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True,kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),bias_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),activity_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001),recurrent_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001)),name='bilstm')(x)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='linear')(attention_mul)
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model

# biGRU
def attention_model_v3(maxlen=21, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )
    #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01) 참고 사이트
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-keras.md
    x = Conv1D(filters = units, kernel_size = 1, activation=keras.activations.elu,strides=30,kernel_regularizer=regularizers.l2(l2=0.0001),bias_regularizer=regularizers.l2(l2=0.0001),activity_regularizer=regularizers.l2(l2=0.0001))(input_layer)
    #x = Dropout(dropout)(x)
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.999,epsilon=1e-06)(x)  # BatchNormalization 대신 LayerNormalization 사용하면?
    x = MaxPooling1D(pool_size=1,strides=2)(x)


    
    lstm_out = tf.keras.layers.Bidirectional(GRU(units, return_sequences=True,kernel_regularizer=regularizers.l2(l2=0.0001),bias_regularizer=regularizers.l2(l2=0.0001),activity_regularizer=regularizers.l2(l2=0.0001),recurrent_regularizer=regularizers.l2(l2=0.0001)),name='bilstm')(x)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='linear')(attention_mul)
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 
    return model

# Bidirectional LSTM + GRU + LSTM + cnn +BiLSTM_attention_model
# 2022-10-31 Written by SEONGJAE-YOO (Commits on Oct 31, 2022)
def Create_BiLSTM_GRU_LSTM_cnn_BiLSTM_attention_model(maxlen=21, units=21, dropout=0.5, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    
    input_layer = Input(shape=(maxlen, n_steps), )
   
    x = tf.keras.layers.Bidirectional(GRU(units,return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)))(input_layer) 



    x =Conv1D(filters=units, kernel_size=1, strides=30,kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-3))(x)
    x = Activation('relu')(x)                        
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    x = MaxPooling1D(pool_size=1,strides=2)(x)

    x = Conv1D(filters=units, kernel_size=1, strides=30,kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-3))(x)
    x = Activation('relu')(x)                        
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    x = MaxPooling1D(pool_size=1,strides=2)(x)

    x = Conv1D(filters=units, kernel_size=1, strides=30,kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-3))(x)
    x = Activation('relu')(x) 
    x = tf.keras.layers.BatchNormalization(axis=1,momentum=0.9)(x) 
    x = MaxPooling1D(pool_size=1,strides=2)(x)



    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True),name='bilstm')(x)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    #attention_mul = Flatten()(attention_mul)


    x_a = GlobalMaxPool1D()(attention_mul)
    x_b = GlobalAveragePooling1D()(attention_mul)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    x = Dense(16)(x) # 16이 적당함
    #x = PReLU()(x) 
    
    output_layer = Dense(1, activation="linear")(x)
    model = Model(inputs=input_layer, outputs=output_layer,name='BidirectionalLSTM_GRU_LSTM_cnn_BiLSTM_attention_model')
    model.summary()  
    
    
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model   

def Create_BiLSTM_GRU_LSTM_cnn_BiLSTM_attention_model_v2(maxlen=21, units=32, dropout=0.3, n_steps=5, LOSS = "mae", optimizer= 'cos'):
    
    input_layer = Input(shape=(maxlen, n_steps), )
   
    x = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout))(input_layer)
    x = TimeDistributed(DropConnectDense(units=128, prob=0.2, activation="relu", use_bias=True))(x)

    x = GRU(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(x)

    x = Dropout(dropout)(x)

    x = LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout)(x)  

    x = Dropout(dropout)(x)      

    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=1)(x)



    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True),name='bilstm')(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    #attention_mul = Flatten()(attention_mul)


    x_a = GlobalMaxPool1D()(attention_mul)
    x_b = GlobalAveragePooling1D()(attention_mul)
    #x_c = AttentionWeightedAverage()(x)
    #x_a = MaxPooling1D(pool_size=2)(x)
    #x_b = AveragePooling1D(pool_size=2)(x)
    x = concatenate([x_a,x_b])
    #x = Dropout(dropout_rate)(x)
    #x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x) 
    model = Model(inputs=input_layer, outputs=x,name='BidirectionalLSTM_GRU_LSTM_cnn_BiLSTM_attention_model_v2')
    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model       

#### TCN_lstm 

def tc_lstm(maxlen=21, units=254, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):

    input_layer = Input(shape=(maxlen, n_steps), )

    o = TCN(units,return_sequences=True,activation='relu',kernel_initializer="he_uniform",dropout_rate=dropout)(input_layer)  # The TCN layers are here.
    #regression=True
    # o = tf.keras.layers.Bidirectional(LSTM(16, return_sequences=True, dropout=dropout,
    #                     recurrent_dropout=dropout, activation='relu',kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(o)
    o = tf.keras.layers.Bidirectional(LSTM(254, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout, activation='tanh',kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)))(o)
    o = Dropout(dropout)(o)  
   # o = Flatten()(o)
    x_a = GlobalMaxPool1D()(o)
    x_b = GlobalAveragePooling1D()(o)
    o = concatenate([x_a,x_b])
    

    o = Dense(1)(o) 
    model = Model(inputs=input_layer, outputs=o,name='BIlstm_TCN')

    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model   

def tc_lstm_v2(maxlen=21, units=10, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):

    input_layer = Input(shape=(maxlen, n_steps), )
    # o = tf.keras.layers.Bidirectional(LSTM(16, return_sequences=True, dropout=dropout,
    #                     recurrent_dropout=dropout, activation='relu',kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(o)
    o = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout, activation='tanh',kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)))(input_layer)
    o = Dropout(dropout)(o)  
    #if return_sequences=False: 2D tensor with shape (batch_size, nb_filters).
    o = TCN(units,return_sequences=True,activation='relu',kernel_initializer="he_uniform",dropout_rate=dropout)(o)  # The TCN layers are here.
    #regression=True
   # o = Flatten()(o)
    x_a = GlobalMaxPool1D()(o)
    x_b = GlobalAveragePooling1D()(o)
    o = concatenate([x_a,x_b])
    

    o = Dense(1)(o) 
    model = Model(inputs=input_layer, outputs=o,name='BIlstm_TCN')

    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model       

# #maxlen= 29 

# In addition,
# regularisation techniques such as Dropout [35] and Batch Normalization [18]
# have been developed to reduce over-fitting, resulting in better predictions when
# applied to out-of-sample data  = >dropout_rate=dropout,use_batch_norm=True
def tc_lstm_v2_maxlen_29(maxlen=29, units=30, dropout=0.5, n_steps=1, LOSS = "mae", optimizer= 'cos'):

    input_layer = Input(shape=(maxlen, n_steps), )
    o = TCN(units,return_sequences=True,activation='relu',kernel_initializer="he_uniform",dropout_rate=dropout,use_batch_norm=True)(input_layer)  # The TCN layers are here.
    #regression=True
    # o = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True, dropout=dropout,
    #                        recurrent_dropout=dropout, activation='tanh',kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)))(o)
    # o = Dropout(dropout)(o)  
    #if return_sequences=False: 2D tensor with shape (batch_size, nb_filters).
   
    x_a = GlobalMaxPool1D()(o)
    x_b = GlobalAveragePooling1D()(o)
    o = concatenate([x_a,x_b])
    

    o = Dense(1)(o) 
    model = Model(inputs=input_layer, outputs=o,name='TCN')

    model.summary()  
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])

    return model           

#####

def Create_Bidirectional_GRU_LSTM_v2(maxlen=21, units=256, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
    
    X_shortcut1 = Input(shape=(maxlen, n_steps), )

   # X_shortcut1 = input_layer 
    X = Conv1D(filters=units, kernel_size=1, strides=30,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X_shortcut1)         
    X = Activation('relu')(X)                        
    X = Conv1D(filters=units, kernel_size=1, strides=30,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X)         
    X = Activation('relu')(X)                        

    # connect shortcut to the main path
    X = Activation('relu')(X_shortcut1)  # pre activation
    X = Add()([X_shortcut1,X])
    X = MaxPooling1D(pool_size=1, strides=2, padding='same')(X)

    # second block
    X_shortcut2 = X
    X = Conv1D(filters=units, kernel_size=1, strides=15,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X)

    # connect shortcut to the main path
    X = Activation('relu')(X)                        

    X = Conv1D(filters=units, kernel_size=1, strides=15,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X)
    X = Activation('relu')(X)                        
    
    X = Activation('relu')(X_shortcut2)  # pre activation                       
    X = Add()([X_shortcut2,X])
    X = MaxPooling1D(pool_size=1, strides=2, padding='same')(X)
    
    X = tf.keras.layers.Bidirectional(GRU(units,return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.001)))(X) 
    
   
    
    # X = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=dropout,
    #                        recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(X) 
    # X = Activation('relu')(X)
    # X = Dropout(dropout)(X)

    # X = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=dropout,
    #                        recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(X) 
    # X = Activation('relu')(X)
    # X = Dropout(dropout)(X)
    
    

    x_a = GlobalMaxPool1D()(X)
    x_b = GlobalAveragePooling1D()(X)
    X = concatenate([x_a,x_b])
   

    # X = Dense(16)(X) # 16이 적당함
    # X= PReLU()(X) 
    
    output_layer = Dense(1)(X)

    model = tf.keras.models.Model(inputs=X_shortcut1, outputs=output_layer)
    model.summary()
    # plot_model(model, to_file='ai\modelPicture\Bidirectional_GRU_LSTM_v2_model.png', show_shapes=True, 
    #            show_layer_names=True,
    #            rankdir='TB',
    #            expand_nested=True,
    #            dpi=96)
    #create your model
    #then call the function on your model
    visualize_model(model)

    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    return model    

# BatchNormalization 추가할때 참고한 사이트
# 참고 - https://buomsoo-kim.github.io/keras/2018/05/05/Easy-deep-learning-with-Keras-11.md/

def Create_Bidirectional_GRU_LSTM_v3(maxlen=21, units=32, dropout=0.3, n_steps=5, LOSS = "mae", optimizer= 'cos'):
    
    X_shortcut1 = Input(shape=(maxlen, n_steps), )

   # X_shortcut1 = input_layer 
    X = Conv1D(filters=units, kernel_size=1, strides=30,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X_shortcut1)     
    X = tf.keras.layers.BatchNormalization(axis=1,momentum=0.999)(X)    # axis=1 =>2차원?? 성능 좋을까?                        
    X = Activation('relu')(X)                        
    X = Conv1D(filters=units, kernel_size=1, strides=30,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X)         
    X = tf.keras.layers.BatchNormalization(axis=1,momentum=0.999)(X)
    X = Activation('relu')(X)                        

    # connect shortcut to the main path
    X = Activation('relu')(X_shortcut1)  # pre activation
    X = Add()([X_shortcut1,X])
    X = MaxPooling1D(pool_size=1, strides=2, padding='same')(X)

    # second block
    X_shortcut2 = X
    X = Conv1D(filters=units, kernel_size=1, strides=15,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X)
    X = tf.keras.layers.BatchNormalization(axis=1,momentum=0.999)(X)
    # connect shortcut to the main path
    X = Activation('relu')(X)                        

    X = Conv1D(filters=units, kernel_size=1, strides=15,padding='same',kernel_initializer="he_uniform",
                           kernel_regularizer=regularizers.l2(1e-4))(X)
    X = tf.keras.layers.BatchNormalization(axis=1,momentum=0.999)(X)
    X = Activation('relu')(X)                        
    
    X = Activation('relu')(X_shortcut2)  # pre activation                       
    X = Add()([X_shortcut2,X])
    X = MaxPooling1D(pool_size=1, strides=2, padding='same')(X)
    
    X = tf.keras.layers.Bidirectional(GRU(units,return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)))(X) 
    
   
    X = Activation('relu')(X) 
   
    # X = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=dropout,
    #                        recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(X) 
    # X = Activation('relu')(X)
    # X = Dropout(dropout)(X)

    # X = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=dropout,
    #                        recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(X) 
    # X = Activation('relu')(X)
    # X = Dropout(dropout)(X)
    
    

    x_a = GlobalMaxPool1D()(X)
    x_b = GlobalAveragePooling1D()(X)
    X = concatenate([x_a,x_b])
   

    X = Dense(16)(X) # 16이 적당함
    #X = tf.keras.layers.BatchNormalization(momentum=0.99)(X)
    X= PReLU()(X) 
    
    
    output_layer = Dense(1, activation="linear")(X)

    model = tf.keras.models.Model(inputs=X_shortcut1, outputs=output_layer)
    model.summary()
    # plot_model(model, to_file='ai\modelPicture\Bidirectional_GRU_LSTM_v2_model.png', show_shapes=True, 
    #            show_layer_names=True,
    #            rankdir='TB',
    #            expand_nested=True,
    #            dpi=96)
    #create your model
    #then call the function on your model
    visualize_model(model)

    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
                    
    #model.compile(optimizer='rmsprop', loss='mse', metrics=[r2, mae, mse, rmse, mape, rmsle, nrmse])
                
    return model        




def create_model_bidirectional_v4( n_steps=1,maxlen=21, units=21, cell=LSTM, n_layers=5, dropout=0,
               LOSS = "mae", optimizer="cos", bidirectional=True):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True,activation="tanh"), input_shape=(maxlen, n_steps)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(maxlen, n_steps)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=False,activation="tanh")))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True, dropout=0.5,
                           recurrent_dropout=0.5,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001),activation="tanh")))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 

    return model        

def create_model_bidirectional_GRU_v4( n_steps=1,maxlen=21, units=21, cell=GRU, n_layers=2, dropout=0.7,
               LOSS = "mae", optimizer="cos", bidirectional=True):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True,dropout=dropout,
                           recurrent_dropout=dropout,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001),activation="tanh"), input_shape=(maxlen, n_steps)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(maxlen, n_steps)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=False,dropout=dropout,
                           recurrent_dropout=dropout,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001),activation="tanh")))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001),activation="tanh")))
                model.add(TimeDistributed(DropConnectDense(units=1, prob=0.2, activation="tanh", use_bias=True)))
               # model.add(TimeDistributed(Dense(1)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1)) #,activation='linear'
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae',tf.keras.metrics.RootMeanSquaredError()]) 

    return model            

# 0.019781141 (mae value), units=32
def create_model_lstm_basic( n_steps=1,maxlen=21, LOSS = "mae", optimizer="cos"):
    model = Sequential()
# sequential이니까 순차적으로 쌓인다
    model.add(LSTM(units=32 ,input_shape=(maxlen, n_steps)))
# Dense : 출력층 값이 1개가 나온다. 우리가 예측한 주가, 이것을 통해서 오차를 구하고 학습을 해서 모델을 만드는 것
    model.add(Dense(units=1))
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]) 

    return model       

#units=21
def create_model_lstm_basic_units_21( n_steps=1,maxlen=21, LOSS = "mae", optimizer="cos"):
    model = Sequential()
# sequential이니까 순차적으로 쌓인다
    model.add(LSTM(units=21 ,input_shape=(maxlen, n_steps)))
# Dense : 출력층 값이 1개가 나온다. 우리가 예측한 주가, 이것을 통해서 오차를 구하고 학습을 해서 모델을 만드는 것
    model.add(Dense(units=1))
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]) 

    return model           
     

def create_model_lstm_basic_5( n_steps=1,maxlen=21, LOSS = "mae", optimizer="cos"):
    

    input_layer = Input(shape=(maxlen, n_steps), )

    conv = Conv1D(filters = 32, kernel_size = 1, activation='relu')(input_layer)
    conv = MaxPooling1D(pool_size=1)(conv)
    conv1 = Conv1D(filters = 32, kernel_size = 1, activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=1)(conv1)
    conv2 = Conv1D(filters = 32, kernel_size = 1, activation='relu')(conv1)
    conv3 = Conv1D(filters = 32, kernel_size = 1, activation='relu')(conv2)
    conv4 = Conv1D(filters = 32, kernel_size = 1, activation='relu')(conv3)
    conv5 = Conv1D(filters = 32, kernel_size = 1, activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=1)(conv5)

    z = Dropout(0.5)(Dense(32)(conv5))
    #x = GlobalMaxPool1D()(x)
    x = LSTM(32)(z)
    #x = Flatten()(x)
    x = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()
    model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]) 

    return model     





#1104 Transformer_model
def create_Transformer_model(n_steps=1,maxlen=21,d_k=256,d_v = 256,n_heads = 12,ff_dim=256,LOSS = "mae", optimizer= 'cos'):
  '''Initialize time and transformer layers'''
  time_embedding = Time2Vector(n_steps)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
#   attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
#   attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

  '''Construct model'''
  in_seq = Input(shape=(maxlen,n_steps))  
  x = time_embedding(in_seq)
  x = Concatenate(axis=-2)([in_seq, x])
  x = attn_layer1((x, x, x))
#   x = attn_layer2((x, x, x))
#   x = attn_layer3((x, x, x))
  x = GlobalAveragePooling1D(data_format='channels_last')(x)
  x = Dropout(0.3)(x)
  #x = Dense(16, activation='tanh')(x)
  x = Dense(64, activation='tanh')(x)
  x = Dropout(0.3)(x)
  out = Dense(1)(x)

  model = Model(inputs=in_seq, outputs=out)
  model.summary()
    
  visualize_model(model)
#   model.compile(loss=tf.keras.losses.Huber(), 
#                     optimizer=AngularGrad(optimizer), 
#                     metrics=['mae','mse'])
  model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]) 
                

  return model

#1106 
def create_Transformer_model_v2(n_steps=1,maxlen=5,d_k=50,d_v = 50,n_heads = 12,ff_dim=50,LOSS = "huber_loss",optimizer= 'cos'):
  
  
  
  '''Initialize time and transformer layers'''
  time_embedding = Time2Vector(n_steps)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

  '''Construct model'''
  in_seq = Input(shape=(maxlen,n_steps))  
  x = time_embedding(in_seq)
  x = Concatenate(axis=-2)([in_seq, x])
  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  x = attn_layer3((x, x, x))
  
  x = tf.keras.layers.Bidirectional(GRU(124,return_sequences=True, dropout=0.5,
                           recurrent_dropout=0.5,kernel_regularizer=l2(0.0001),recurrent_regularizer=l2(0.0001)))(x) 
  
  x = GlobalAveragePooling1D(data_format='channels_last')(x)
  x = Dropout(0.3)(x)
  
  x = Dense(16, activation='relu')(x)
  # x = Dense(64, activation='tanh')(x)
  x = Dropout(0.3)(x)
  out = Dense(1, activation='linear')(x)

  model = Model(inputs=in_seq, outputs=out)
  model.summary()
    
  visualize_model(model)
#   model.compile(loss=tf.keras.losses.Huber(), 
#                     optimizer=AngularGrad(optimizer), 
#                     metrics=['mae','mse'])
  model.compile(loss=LOSS, 
                optimizer=AngularGrad(optimizer),  
                metrics=['mae', 'mse', 'mape'])

  return model







# def Create_Bidirectional_GRU_LSTM_v2(maxlen=5, units=128, dropout=0.6, n_steps=100, loss = 'mae', optimizer= 'cos'):
    
#     input_layer = Input(shape=(maxlen, n_steps), )

#     x = Conv1D(filters=units, kernel_size=1, padding='same',kernel_initializer="he_uniform",
#                           kernel_regularizer=regularizers.l2(1e-3))(input_layer)
#     x = PReLU()(x)                    
#     x = MaxPooling1D(pool_size=1)(x)
#     x = Dropout(dropout)(x)

#     x = tf.keras.layers.Bidirectional(GRU(units, return_sequences=True, dropout=dropout,
#                            recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(x)
#     # x = PReLU()(x)                       
#     # x = Dropout(dropout)(x)
#     # x = tf.keras.layers.Bidirectional(GRU(units,return_sequences=True, dropout=dropout,
#     #                        recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(x)
#     x = PReLU()(x)                        
#     x = Dropout(dropout)(x)
#     x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True, dropout=dropout,
#                            recurrent_dropout=dropout,kernel_regularizer=l2(0.001),recurrent_regularizer=l2(0.001)))(x)
#     x = PReLU()(x)                       
#     x = Dropout(dropout)(x)
#     # conv1d_out= TimeDistributed(Conv1D(kernel_size=1, filters=units, padding='same',activation='tanh'))(x)
#     # maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
#     x = attention_3d_block2(x)
    
#     x_a = GlobalMaxPool1D()(x)
#     x_b = GlobalAveragePooling1D()(x)
#     x = concatenate([x_a,x_b])
#     x = PReLU()(x)                        
#     x = Dropout(dropout)(x)

#     x = Dense(16)(x) # 16이 적당함
#     x = PReLU()(x) 
    
#     output_layer = Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.summary()
#     model.compile(loss=loss, 
#                     optimizer=AngularGrad(optimizer), 
#                     metrics=[loss])
#     return model        

#####

# def Create_BiLSTM_GRU_LSTM_cnn_BiLSTM_attention_model_v3(maxlen=7, units=128, dropout=0.2, n_steps=100, loss = "mae", optimizer= 'cos'):
    
#     input_layer = Input(shape=(maxlen, n_steps), )
   
    
#     x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(input_layer)
#     x = MaxPooling1D(pool_size=1)(x)
#     # x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
#     # x = MaxPooling1D(pool_size=1)(x)
#     # x = Conv1D(filters=units, kernel_size=1, padding='same', activation='relu')(x)
#     # x = MaxPooling1D(pool_size=1)(x)
#     x = Dropout(0.7)(x)
#     x = tf.keras.layers.Permute((2, 1))(x)
#     x = attention_3d_block2(x)
#     x = tf.keras.layers.Bidirectional(GRU(units, return_sequences=True),name='biGRU')(x)
#     #x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
#     x = Dropout(0.7)(x)
#     x = attention_3d_block2(x)
#     #x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
#     x = Dropout(0.7)(x)


#     x_a = GlobalMaxPool1D()(x)
#     x_b = GlobalAveragePooling1D()(x)
#     #x_c = AttentionWeightedAverage()(x)
#     #x_a = MaxPooling1D(pool_size=2)(x)
#     #x_b = AveragePooling1D(pool_size=2)(x)
#     x = concatenate([x_a,x_b])


#     x = Dropout(0.7)(x)
 

  
#     x = Dense(1, activation="sigmoid")(x) 
#     model = Model(inputs=input_layer, outputs=x,name='BidirectionalLSTM_GRU_LSTM_cnn_BiLSTM_attention_model_v3')
#     model.summary()  
#     model.compile(loss=loss, 
#                     optimizer=AngularGrad(optimizer), 
#                     metrics=[loss])

#     return model           
# #####
# def create_conv1(units=50, dropout=0.3, n_steps=100, loss='mae', optimizer='adam', n_layers=4, cell='conv1'):

#     model = Sequential()

#     model.add(Conv1D(256, kernel_size=1,padding='same', activation='relu', input_shape=(None, n_steps)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))

#     model.add(Conv1D(128, kernel_size=1, padding='same',activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))

#     model.add(Conv1D(64, kernel_size=1, padding='same',activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))

#     model.add(Conv1D(32, kernel_size=1, padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=1, strides=2, padding='same'))

#     x_a = GlobalMaxPool1D()(model)
#     x_b = GlobalAveragePooling1D()(model)
#     model = concatenate([x_a,x_b])
    
#     #model.add(Flatten())

#     #model.add(Dense(1,kernel_initializer="uniform",activation='relu'))
#     model.add(Dense(1,activation='relu'))
#     model.add(Dense(1,activation='sigmoid'))

#     model.compile(loss='mae', 
#                 optimizer=AngularGrad('cos'), 
#                 metrics=[tf.keras.metrics.MeanSquaredError()])
#     model.summary()

#     return model

# 그래프 출력 함수


def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))

    plt.plot(y_test[-200000:200000], c='b')
    plt.plot(y_pred[-200000:200000], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    
    plt.show()




    # plt.plot(y_pred.ravel(), 'r-', label = 'Predicted Price')
    # # y_test는 실제 값
    # plt.plot(y_test.ravel(), 'b-', label = 'Actual Price')
    # plt.plot((y_pred-y_test).ravel(), 'g-', label = 'diff*10')



# # 위에서 만든 모델로 예측 (3차원 데이터를 넣어줘야함)
# pred_y = model.predict(X_test)


# plt.figure(figsize=[15,6])
# # ravel() 1차원으로 변경
# # pred_y는 예측한 값
# plt.plot(pred_y.ravel(), 'r-', label = 'pred_y')
# # y_test는 실제 값
# plt.plot(y_test.ravel(), 'b-', label = 'y_test')
# # plt.plot((pred_y-y_test).ravel(), 'g-', label = 'diff*10')

# plt.legend() # 범례 표시
# plt.title("samsung")
# plt.show()

# # history : 학습한 history를 저장하고 있음
# plt.plot(h.history['loss'], label = 'loss')
# plt.legend()
# plt.title('Loss')
# # x축이 epochs / y축이 loss
# plt.show()






'''
    #https://github.com/hungchun-lin/Stock-price-prediction-using-GAN/blob/master/Code/3.%20Baseline_GRU.py    
    pyplot.plot(history['loss'], label='train')
    pyplot.plot(history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    ____________
     plt.plot(pd.DataFrame(history.history()))
    plt.grid(True)
    plt.show()

'''

# 참고 -https://github.com/XifengGuo/CapsNet-Keras/issues/69
def visualize_model(model):
  return SVG(model_to_dot(model, show_shapes=True,show_layer_names= True,rankdir = 'TB',expand_nested=True,dpi=44).create(prog='dot', format='svg'))
# (model: Any, show_shapes: bool = False, show_layer_names: bool = True, rankdir: str = 'TB', expand_nested: bool = False, dpi: int = 96, subgraph: bool = False) -> Any


# 참고 -https://colab.research.google.com/drive/1L1STGmVK5IgdjLpEb-o8tuJ0yPCZ65Mt?usp=sharing#scrollTo=AMAqaORfirvJ
# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function
def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return func_value

# CS - Parameters
parameters = {
    'birds': 500,
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'iterations': 250,
    'discovery_rate': 0.25,
    'alpha_value': 0.01,
    'lambda_value': 1.5,
    'verbose': True
}    