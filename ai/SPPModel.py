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
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout , Activation, GRU

from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
#https://github.com/linewalks/AngularGrad-tf 참고 
from .angular_grad import AngularGrad
from matplotlib import pyplot
####
from keras.layers import GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation, Add, Flatten, BatchNormalization , GlobalAveragePooling1D
from keras.layers import Dense, Embedding, Input, concatenate

from .attention_3d_block.attention_3d_block import attention_3d_block2

plt.rcParams['font.family'] = 'Malgun Gothic'


class DataNotEnough(BaseException):
    pass

# 학습 함수
def train(data, model, n_epochs=400, batch_size=64, verbose=1):
    early_stopping = EarlyStopping(monitor='val_loss', patience=200)  # 200번이상 더 좋은 결과가 없으면 학습을 멈춤

    # verbose 옵션은 실행 과정을 콘솔에 띄워줄지 말지에 대한 옵션
    # 0 - 끔, 1 - 움직이는 실시간 그래프, 2 - 정적 메시지
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[early_stopping],
                        verbose=verbose)
    
    return history

# 에러 평가 함수
def evaluate(data, model):
    mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # 스케일링된 결과 값을 본래 값으로 복원한다 (inverse_transform 함수란?)
    mean_absolute_error = data["column_scaler"]["close"].inverse_transform([[mae]])[0][0]
    return mean_absolute_error


# 예측 주가를 계산 해주는 함수
def predict(data, model, n_steps=100):
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
def load_data(df, n_steps=100, lookup_step=10, test_size=0.3, shuffle=True):
    # return 해줘야 할 모든 값들은 result 변수에 넣을 예정
    result = {}

    column_scaler = {}
    # data를 칼럼별로 0과 1사이의 값으로 scale
    for column in df.columns:  # close, volume, open, high, low 컬럼들을 모두 MinMaxScaler 해준다.
        scaler = MinMaxScaler() # scaler = StandardScaler()
        #scaler = tf.keras.utils.normalize(column_scaler, axis=1)
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
def create_model(units=50, dropout=0.3, n_steps=100, loss='mae', optimizer='cos', n_layers=4, cell=LSTM):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(cell(units, return_sequences=True, input_shape=(None, n_steps)))
        elif i == n_layers - 1:  # 마지막 layer
            model.add(cell(units))
        else:
            model.add(cell(units, return_sequences=True))
        # 매 layer마다 dropout을 해줌
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    model.summary()

    return model

#A bidirectional long short-term memory (BiLSTM) network 
# is a combination of two LSTMs, i.e., forward and backward.
# Based on LSTM, BiLSTM can extract the feature of forward and backward simultaneously
# 2022-10-25 Written by SEONGJAE-YOO (Commits on Oct 25, 2022)
def create_model_Bidirectional(units=128, dropout=0.5, n_steps=100, loss = 'mae', optimizer= 'cos', n_layers=4, cell=LSTM):
    #model = Sequential()
    # for i in range(n_layers):
    #     if i == 0:
    #         model.add(cell(units, return_sequences=True, input_shape=(None, n_steps)))
    #     elif i == n_layers - 1:  # 마지막 layer
    #         model.add(cell(units))
    #     else:
    #         model.add(cell(units, return_sequences=True))
    #     # 매 layer마다 dropout을 해줌
    #     model.add(Dropout(dropout))
    # model.add(Dense(1))
    # model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    
    # model.add(Bidirectional(LSTM(10, return_sequences=True),
    #                          input_shape=(5, 10)))
    # model.add(Bidirectional(LSTM(10)))
    
    model = Sequential()
    for i in range(n_layers):
            if i ==0:
                model.add(tf.keras.layers.Bidirectional(cell(128,return_sequences=True, activation="relu") , input_shape=(None, n_steps)))          
            elif i == n_layers - 1:  # 마지막 layer
                model.add(tf.keras.layers.Bidirectional(cell(64, return_sequences=False,activation="relu")))
            elif i == n_layers - 2:  # 마지막 2번째 layer
                model.add(tf.keras.layers.Bidirectional(cell(128,return_sequences=True, activation="relu")))    
            else:
                model.add(tf.keras.layers.Bidirectional(cell(256,return_sequences=True,activation="relu")))
            # 매 layer마다 dropout을 해줌
            model.add(Dropout(dropout))
    model.add(tf.keras.layers.Dense(5, activation='relu')) # tf.keras.layers.Activation(tf.nn.relu)
    #model.add(activation ='softmax')  # model.add(layers.Dense(64, activation='relu'))
    #model.compile(loss = tf.keras.losses.CategoricalCrossentropy() , optimizer= tf.keras.optimizers.RMSprop(learning_rate=1e-3))
    model.compile(loss=loss, metrics=[tf.keras.metrics.MeanSquaredError()], optimizer=AngularGrad(optimizer))
  #  model.compile(loss=loss, metrics=[tf.keras.metrics.MeanSquaredError()], optimizer=optimizer)
    model.summary()

    # model = Sequential()
    # #layer = tf.keras.layers.Activation('softmax')
    # for i in range(n_layers):
    #         if i ==0:
    #             model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True, activation="relu") , input_shape=(7 , n_steps)))          
    #         elif i == n_layers - 1:  # 마지막 layer
    #             model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=False, activation="relu")))
    #         else:
    #             model.add(tf.keras.layers.Bidirectional(cell(units, return_sequences=True, activation="relu")))
    #         # 매 layer마다 dropout을 해줌
    #         model.add(Dropout(dropout))
    # model.add(tf.keras.layers.Dense(1)) # tf.keras.layers.Activation(tf.nn.relu)
    # #model.add(activation ='softmax')  # model.add(layers.Dense(64, activation='relu'))
    # #model.compile(loss = tf.keras.losses.CategoricalCrossentropy() , optimizer= tf.keras.optimizers.RMSprop(learning_rate=1e-3))
    # model.compile(loss=loss, metrics=[loss], optimizer=optimizer)
    # model.summary()
      
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
def create_lstm_cnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    input_layer = Input(shape=(maxlen,embed_size))
    #input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix],
    #              trainable=False)(inp)
    x = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(input_layer)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters=recurrent_units, kernel_size=3, padding='same', activation='relu')(x)
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

    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x, name='lstm_cnn')
    model.summary()
    model.compile(loss='mae', 
                optimizer=AngularGrad('cos'), 
                metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


####

# 2022-10-26 Written by SEONGJAE-YOO (Commits on Oct 26, 2022)
# LSTM-CNN-version2
def create_dpcnn(maxlen, embed_size, recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    input_layer = Input(shape=(maxlen,embed_size))
    #input_layer = Input(shape=(maxlen, embed_size), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix],
    #              trainable=False)(inp)
    X_shortcut1 = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate)(input_layer)
    X_shortcut1 = Dropout(dropout_rate)(X_shortcut1)
    # first block
    X_shortcut1 = Conv1D(filters=recurrent_units, kernel_size=1, padding='same',strides=3)(X_shortcut1)
    X_shortcut1 = Activation('relu')(X_shortcut1)
    X_shortcut1 = Conv1D(filters=300, kernel_size=1, padding='valid',strides=3)(X_shortcut1)
    X_shortcut1 = Activation('relu')(X_shortcut1)


    # # connect shortcut to the main path
    # X = Activation('relu')(input_layer)  # pre activation
    # X = Add()([X,X_shortcut1])
    X_shortcut1 = MaxPooling1D(pool_size=1, strides=2, padding='valid')(X_shortcut1)


    # second block
    X_shortcut2 = X_shortcut1
    X_shortcut2 = Conv1D(filters=recurrent_units, kernel_size=1, strides=3)(X_shortcut2)
    X_shortcut2 = Activation('relu')(X_shortcut2)
    X_shortcut2 = Conv1D(filters=recurrent_units, kernel_size=1, strides=3)(X_shortcut2)
    X_shortcut2 = Activation('relu')(X_shortcut2)

    # connect shortcut to the main path
    X_shortcut2 = MaxPooling1D(pool_size=1, strides=2, padding='valid')(X_shortcut2)

    # Output
    #X = Flatten()(X)

    x_a = GlobalMaxPool1D()(X_shortcut2)
    x_b = GlobalAveragePooling1D()(X_shortcut2)
    X  = concatenate([x_a,x_b])

    X = Dense(nb_classes,kernel_initializer="uniform",activation='sigmoid')(X)

    model = Model(inputs = input_layer, outputs = X, name='dpcnn')
    model.summary()
    model.compile(loss='mae', 
                optimizer=AngularGrad('cos'), 
                metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

# 2022-10-26 Written by SEONGJAE-YOO (Commits on Oct 26, 2022)
#####GRU-CNN
## cnn3 
def create_GRU_CNN(maxlen=5, units=64,recurrent_dropout_rate=0.3 ,dropout=0.3, n_steps=100, loss = 'mae', optimizer= 'cos'):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(maxlen, n_steps), )
    x = GRU(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=recurrent_dropout_rate)(input_layer)
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
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x,name='GRU-CNN')
    model.summary()  
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

######
# 2022-10-26 Written by SEONGJAE-YOO (Commits on Oct 26, 2022)
# CNN_GRU 
def create_cnn_GRU(maxlen=5, units=128, dropout=0.5, n_steps=390, loss = 'mae', optimizer= 'cos'):

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
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x, name ='CNN_GRU')
    model.summary()  
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[tf.keras.metrics.MeanSquaredError()])
                
    return model

####
# Bidirectional GRU + Bidirectional LSTM
def Create_Bidirectional_GRU_LSTM(maxlen=5, units=64, dropout=0.3, n_steps=390, loss = 'mae', optimizer= 'cos'):
    
    input_layer = Input(shape=(maxlen, n_steps), )
   
    x = tf.keras.layers.Bidirectional(GRU(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout))(input_layer)
    x = Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True, dropout=dropout,
                           recurrent_dropout=dropout))(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a,x_b])

    x = Dense(32, activation="relu")(x)
    output_layer = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

#### 2022-10-28 Written by SEONGJAE-YOO (Commits on Oct 28, 2022)
# Bidirectional LSTM + GRU + LSTM_ cnn
def Create_BidirectionalLSTM_GRU_LSTM(maxlen=5, units=200, dropout=0.2, n_steps=100, loss = 'mae', optimizer= 'cos'):
    
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
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x) # softmax를 사용하면 어떨까?
    model = Model(inputs=input_layer, outputs=x,name='BidirectionalLSTM_GRU_LSTM_CNN')
    model.summary()  
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[tf.keras.metrics.MeanSquaredError()])

    return model    


# filter_kernels 적용된 cnn
def create_filter_kernels_conv(maxlen=5, units=256, dropout=0.5, n_steps=100, loss = 'mae', optimizer= 'cos'):
    #filter_kernels = [7, 7, 5, 5, 3, 3]
    input_layer = Input(shape=(maxlen, n_steps), )

    conv = Conv1D(nb_filter=units, filter_length=7, border_mode='valid', activation='relu')(input_layer)
    conv = MaxPooling1D(pool_length=3)(conv)
    conv1 = Conv1D(nb_filter=units, filter_length=7, border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)
    conv2 = Conv1D(nb_filter=units, filter_length=5, border_mode='valid', activation='relu')(conv1)
    conv3 = Conv1D(nb_filter=units, filter_length=5, border_mode='valid', activation='relu')(conv2)
    conv4 = Conv1D(nb_filter=units, filter_length=3, border_mode='valid', activation='relu')(conv3)
    conv5 = Conv1D(nb_filter=units, filter_length=3, border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)
    z = Dropout(dropout)(Dense(64, activation='relu')(conv5))
    #x = GlobalMaxPool1D()(x)
    x = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()  
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

#### CNN-BiLSTM-Attention model


def attention_model(maxlen=5, units=64, dropout=0.3, n_steps=100, loss = 'mae', optimizer= 'cos'):
    input_layer = Input(shape=(maxlen, n_steps), )

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(input_layer)  #, padding = 'same'
    x = Dropout(dropout)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_out = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True),name='bilstm')(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[input_layer], outputs=output)
    model.summary()  
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[tf.keras.metrics.MeanSquaredError()])
    return model




# Bidirectional LSTM + GRU + LSTM + cnn +BiLSTM_attention_model
# 2022-10-31 Written by SEONGJAE-YOO (Commits on Oct 31, 2022)
def Create_BiLSTM_GRU_LSTM_cnn_BiLSTM_attention_model(maxlen=5, units=200, dropout=0.2, n_steps=100, loss = "mae", optimizer= 'cos'):
    
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
    x = Dense(1, activation="relu")(x) # softmax를 사용하면 어떨까?
    model = Model(inputs=input_layer, outputs=x,name='BidirectionalLSTM_GRU_LSTM_cnn_BiLSTM_attention_model')
    model.summary()  
    model.compile(loss=loss, 
                    optimizer=AngularGrad(optimizer), 
                    metrics=[loss])

    return model   
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

# def plot_graph(model, data):
#     y_test = data["y_test"]
#     X_test = data["X_test"]
#     y_test = model.predict(X_test)
#     y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
#     y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
#     # 마지막 200개의 데이터를 보여줌. 기간 수정을 원하시면 이 숫자를 바꿔주세요.
#     plt.plot(y_test[-200:], c='b')
#     plt.plot(y_pred[-200:], c='r')
#     plt.xlabel("Days")
#     plt.ylabel("Price")
#     plt.legend(["Actual Price", "Predicted Price"])
#     plt.show()

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