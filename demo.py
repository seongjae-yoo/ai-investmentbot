ver = "#version 0.0.2"
print(f"demo Version: {ver}")

import pymysql
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping #모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수입니다.

from tensorflow.keras.layers import LSTM,GRU


from ai.SPPModel import load_data, DataNotEnough, predict, train , LSTM_CNN, Deep_CNN, CNN_GRU, BiGRU_BiLSTM, BiLSTM_GRU_LSTM_CNN ,CNN_Version2, CNN_BiLSTM_Attention, BiGRU_CNN_BiLSTM_Attention, BiLSTM_GRU_LSTM_CNN_BiLSTM_attention,TCN_BiLSTM, plot_graph, Create_Bidirectional_GRU_LSTM_v3, create_Transformer_model, create_Transformer_model_v2, CNN_Attention, create_model_bidirectional_v4,create_model_lstm_basic, GRU_CNN, BiLSTM_TCN, TCN, create_model_bidirectional_GRU_v4,CNN_BiGRU_Attention,BiLSTM_Attention_CNN , CNN_Attention_BiLSTM, LSTM_layers_4_v2, LSTM_layers_4 , Bi_LSTM_layers_4, CNN_Attention_BiLSTM_Version2, CNN_Attention_BiLSTM_Version3, CNN_Attention_BiLSTM_Version4,Deep_CNN_BiGRU, BiLSTM_layers_4_Version2, CNN_Attention_BiLSTM_Attention,BiLSTM_Attention, BiLSTM_Attention_sigmoid, CNN_Attention_BiLSTM_Version5, BiLSTM_single_attention_vector,CNN_Attention_BiLSTM_Version6, CNN_Attention_BiLSTM_Version7, CNN_Attention_BiLSTM_Version8, CNN_Attention_BiLSTM_Version9, CNN_Attention_BiLSTM_Version10, CNN_Attention_BiLSTM_Version11, CNN_Attention_BiLSTM_Version12, CNN_Attention_BiLSTM_Version13, CNN_Attention_BiLSTM_Version14, CNN_Attention_BiLSTM_Version15, CNN_Attention_BiLSTM_Version16, CNN_Attention_BiLSTM_Version17, CNN_Attention_BiLSTM_Version18, CNN_Attention_BiLSTM_Version17_test,CNN_Attention_BiLSTM_Version11_version2, CNN_Attention_BiLSTM_Version11_version3,CNN_Attention_BiLSTM_Version17_load_weights, CNN_Attention_BiLSTM_Version19, CNN_Attention_BiLSTM_Version20, CNN_Attention_BiLSTM_Version21, CNN_Attention_BiLSTM_Version22
from library import cf

####2022-11-02
#from ai.SPPModel import evaluate


#### 2022-11-04
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model 

import matplotlib.pyplot as plt

import matplotlib as mpl

import time
import os
from ai.SPPModel import train

#pymysql에 connect라는 함수를 이용하여 db 서버에 접속, 연결 할 수 있다.
conn = pymysql.connect(host=cf.db_ip,
                       port=int(cf.db_port),
                       user=cf.db_id,
                       password=cf.db_passwd,
                       db='daily_craw',
                       charset='utf8mb4',
                       cursorclass=pymysql.cursors.DictCursor)


# 상장시기 부터 전체 데이터로 하면 maxlen이 5개가 밑에 21개보다 mae값이 더 작음 
FEATURE_COLUMNS = ["close", "volume", "open", "high", "low"]
#FEATURE_COLUMNS = ["d1_diff_rate", "clo5_diff_rate", "clo10_diff_rate", "clo20_diff_rate", "clo40_diff_rate"]
#5개 컬럼에서 date 추가하면 성능 매우안좋아짐

#maxlen=21
# FEATURE_COLUMNS   = [ 'close', 'open', 'high', 'low',
#                     'volume', 'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
#                     'clo100', 'clo120','yes_clo5', 'yes_clo10', 'yes_clo20', 'yes_clo40', 'yes_clo60','yes_clo80','yes_clo100', 'yes_clo120'
#                     ] 


# code_name = '동화기업'
# until = '20220504'
# sql = """
#     SELECT {} FROM `{}`
#     WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
# """.format(','.join(FEATURE_COLUMNS), code_name, until) #STR_TO_DATE : 형식 문자열에 날짜 및 시간 부분이 모두 포함 된 경우 DATETIME 값을 반환


#19940502 부터 20221117
#20131106
code_name = '방림'
# 19890530 부터
until = '20230110'

sql = """
    SELECT {} FROM `{}`
    WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
""".format(','.join(FEATURE_COLUMNS), code_name, until) #STR_TO_DATE : 형식 문자열에 날짜 및 시간 부분이 모두 포함 된 경우 DATETIME 값을 반환



df = pd.read_sql(sql, conn)
if not len(df):
    print(f'{code_name}의 {until}까지 데이터가 존재하지 않습니다.')
    exit(1)


# parameters_ 


# maxlen을 큰값으로 잡은 이유 - 모델이 알아서 Output Shape size 데이터크기에 맞게 맞춰줌
# # 다음과 같이 잡아줌 
# WARNING:tensorflow:Model was constructed with shape (None, 100, 100) for input Tensor("input_1:0", shape=(None, 100, 100), dtype=float32), but it was called on an input with incompatible shape (None, 6, 100).

maxlen = 5
#maxlen = 5
# # 하나의 시퀀스에 담을 데이터 수
N_STEPS = 1
#N_STEPS = 5 # 1보다 성능 안좋음 1이 더좋음
# 단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
LOOKUP_STEP = 1 # 5보다 mae 값이 더욱 작게 나옴 
#  train 범위 : test_size 가 0.2 이면 X_train, y_train에 80% 데이터로 트레이닝 하고 X_test,y_test에 나머지 20%로 테스트를 하겠다는 의미
TEST_SIZE = 0.3

# layer 수
#N_LAYERS = 5


#CELL = GRU
# layer의 node수
#UNITS = 128

#recurrent_units = 64

# overfitting 방지를 위해 몇개의 노드를 죽이고 남은 노드들을 통해서만 훈련을 하는 것(0.2 -> 20%를 죽인다)
#DROPOUT = 0.5

# mean absolute error (평균 절대 오차)
#LOSS = "mae"

# 최적화 알고리즘 선택
# 실험결과 adam 보다 AngularGrad cos 가 더 loss 값이 작음
OPTIMIZER = "cos"  
#OPTIMIZER = "adam"

# 각 학습 반복에 사용할 데이터 샘플 수
# 164으로 실험한 결과 loss 값이 70보다 크게 나옴 
BATCH_SIZE = 32

# 학습 횟수
EPOCHS = 100

ratio_cut = 5

is_used_predicted_close =  True  



  

try:
    # shuffle: split을 해주기 이전에 시퀀스를 섞을건지 여부
    shuffled_data = load_data(df=df, n_steps=N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, shuffle=True)
except DataNotEnough:
    print('데이터가 충분하지 않습니다. ')
    exit(1)
  
#모델    y
# model 선택(원하시는 모델 함수를 선택하여 실행해주시면 됩니다.)

#model = LSTM_layers_4_v2()  
model = CNN_Attention_BiLSTM_Version7()
# 학습 시작
history = train(shuffled_data, model, EPOCHS, BATCH_SIZE, verbose=1)

# shuffle 되지 않은 df로 다시 new_df에 저장
new_df = pd.read_sql(sql, conn)
  
data = load_data(df=new_df, n_steps=N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, shuffle=False)


# test_loss,mae,rmse,real_mae,real_rmse = evaluate(data, model)  
# # print(f"train_huber_loss, train_mae, train_rmse: {train_huber_loss,train_mae, train_rmse}")      
# # print(f"test_huber_loss, test_mae, test_rmse: {test_huber_loss,test_mae, test_rmse}")      
# print(f"test_loss,mae,rmse,real_mae,real_rmse: {test_loss,mae,rmse,real_mae,real_rmse}")  
  


future_price = predict(data, model, n_steps=N_STEPS)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}")

plot_graph(model, data)	
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#plot_model(model)
#plot_graph(model, data)
#plot_model(model, to_file='model_shapes.png', show_shapes=True)


# mpl.rcParams['figure.figsize'] = (12, 10)
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# def plot_metrics(history):
#   metrics = ['loss', 'auc', 'precision', 'recall']
#   for n, metric in enumerate(metrics):
#     name = metric.replace("_"," ").capitalize()
#     plt.subplot(2,2,n+1)
#     plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
#     plt.plot(history.epoch, history.history['val_'+metric],
#              color=colors[0], linestyle="--", label='Val')
#     plt.xlabel('Epoch')
#     plt.ylabel(name)
#     if metric == 'loss':
#       plt.ylim([0, plt.ylim()[1]])
#     elif metric == 'auc':
#       plt.ylim([0.8,1])
#     else:
#       plt.ylim([0,1])

#     plt.legend()


# plot_metrics(history, 'loss')