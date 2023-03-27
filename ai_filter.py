# version 0.1.2
import datetime
import sys

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy.event import listen
from sqlalchemy.pool import Pool
from sqlalchemy.exc import InternalError, ProgrammingError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from ai.SPPModel import load_data, predict, DataNotEnough, CNN_Attention_BiLSTM_Version3,CNN_Attention_BiLSTM_Version11, CNN_Attention_BiLSTM_Version17,CNN_Attention_BiLSTM_Version9, CNN_Attention_BiLSTM_Version27




from library import cf
from library.open_api import setup_sql_mod
from sklearn.metrics import mean_absolute_error

import time
import os
# 2022-10-19 Add
import tensorflow as tf  

from tensorflow.keras.layers import LSTM, Dense, Dropout , Activation
from tensorflow.keras.models import Sequential

from wandb.keras import WandbCallback
import wandb


def lr_scheduler(epoch, lr):
    # log the current learning rate onto W&B
    if wandb.run is None:
        raise wandb.Error("You must call wandb.init() before WandbCallback()")

    wandb.log({'learning_rate': lr}, commit=False)
    
    if epoch < 7:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



listen(Pool, 'connect', setup_sql_mod) # 서버와 클라이언트를 연결해준다.
listen(Pool, 'first_connect', setup_sql_mod)


# 모의투자, 실전투자 일때만 들어오는 함수
def filter_by_ai(db_name, simul_num):
    from library.simulator_func_mysql import simulator_func_mysql
    sf = simulator_func_mysql(simul_num, 'real', db_name)
    try:
        ai_filter(sf.ai_filter_num, sf.engine_simulator)
    except AttributeError:
        print(f"{simul_num} 알고리즘은 AI 알고리즘이 아닙니다. \n cf파일에서 simul_num 을 AI알고리즘을 사용하는 번호로 수정해주세요")


def filtered_by_basic_lstm(dataset, ai_settings):
    """
    :param dataset: 실제 주가 데이터
    :param settings: AI 알고리즘 세팅
    :return: ratio_cut(목표 수익률) 보다 ratio가 작으면 True 반환(필터링 대상)
    """

    shuffled_data = load_data(df=dataset.copy(), n_steps=ai_settings['n_steps'], test_size=ai_settings['test_size'])
    
    #!@

    #model = create_model(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                     n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    # model = create_model_Bidirectional(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                         n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    
        
    #model = CNN_Attention_BiLSTM_Version17()                        
    model = ai_settings['model']
    
    #checkpoint_filepath = 'ModelCheckpoint/CNN_Attention_BiLSTM_Version3/Checkpoint'

    #early_stopping = EarlyStopping(monitor='val_loss', patience=500)  # patience 번이상 더 좋은 결과가 없으면 학습을 멈춤
    #callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
    #                                          monitor='val_loss', 
    #                                          save_best_only=True, verbose=1)
    
    #ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, verbose=1, mode='min',monitor='val_loss')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                           patience=5, min_lr=0.001, mode='min',verbose=1)
    #wandb.init(project="simulation_buy", entity="SeongJae-Yoo")
    #wandb.run.name = 'Bi_LSTM_layers_4'
    
    #wandb.run.name = 'test_222'
    # Save a model file manually from the current directory:
    #wandb.save('model-best.h5')

    # restore the model file "model.h5" from a specific run by user "lavanyashukla"
# in project "save_and_restore" from run "10pr4joa"
    #best_model = wandb.restore('model-best_v1.h5', run_path="SeongJae-Yoo/modetour/runs/c2s4ydod")


# use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
    #model.load_weights(best_model.name)
    #model.load_weights('model-best.h5')
    # wandb.log({"gradients": wandb.Histogram(numpy_array_or_sequence)})
    # wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})

    # wandb   Hyperparameter Sweeps   시각화 할 수 있는 자료 (아래 확인)
    #https://colab.research.google.com/drive/1gKixa6hNUB8qrn1CfHirOfTEQm0qLCSS#scrollTo=1gD9qhA9yOYs
    
    #wandb_callback = WandbCallback(monitor='val_loss',save_model=True,mode='min',log_weights=True,log_evaluation=True,validation_steps=5,verbose=1)
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.0001, mode='min',verbose=1)
    
    model.fit(shuffled_data["X_train"], shuffled_data["y_train"],
                        batch_size=ai_settings['batch_size'],
                        epochs=ai_settings['epochs'],
                        validation_data=(shuffled_data["X_test"], shuffled_data["y_test"]),
                        callbacks=[reduce_lr],
                        verbose=1)

    scaled_data = load_data(df=dataset.copy(), n_steps=ai_settings['n_steps'], test_size=ai_settings['test_size'],
                            shuffle=False)

    # result = evaluate(scaled_data, model)
    # print(f"result: {result}")

    
    # train_huber_loss, train_mae, train_rmse,test_huber_loss, test_mae, test_rmse = evaluate(scaled_data, model)  
    # print(f"train_huber_loss, train_mae, train_rmse: {train_huber_loss,train_mae, train_rmse}")      
    # print(f"test_huber_loss, test_mae, test_rmse: {test_huber_loss,test_mae, test_rmse}")      

    # mse = evaluate(scaled_data, model)
    # print(f"Mean Squared Error: {mse}")



    # 예측 가격
    future_price = predict(scaled_data, model, n_steps=ai_settings['n_steps'])

    # 스케일링 된 예측 결과
    scaled_y_pred = model.predict(scaled_data['X_test'])
    # 실제 값으로 변환 된 결과
    y_pred = np.squeeze(scaled_data['column_scaler']['close'].inverse_transform(scaled_y_pred))

    if ai_settings['is_used_predicted_close']:
        close = y_pred[-1] # 예측 그래프에서의 종가
    else:
        close = dataset.iloc[-1]['close'] # 실제 종가

    # ratio : 예상 상승률
    ratio = (future_price - close) / close * 100

    msg = f"After {ai_settings['lookup_step']}: {int(close)} -> {int(future_price)}"

    if ratio > 0: # lookup_step(분, 일) 후 상승 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯅ '
    elif ratio < 0: # lookup_step(분, 일) 후 하락 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯆ '
    print(msg, end=' ')
    return ai_settings['ratio_cut'] >= ratio # ratio_cut(목표 수익률) 보다 ratio가 작으면 True 반환(필터링 대상)



def filtered_by_basic_lstm_v2(dataset, ai_settings):
    """
    :param dataset: 실제 주가 데이터
    :param settings: AI 알고리즘 세팅
    :return: ratio_cut(목표 수익률) 보다 ratio가 작으면 True 반환(필터링 대상)
    """

    shuffled_data = load_data(df=dataset.copy(), n_steps=ai_settings['n_steps'], test_size=ai_settings['test_size'])
    
    #!@

    #model = create_model(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                     n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    # model = create_model_Bidirectional(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                         n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    
        
    #model = CNN_Attention_BiLSTM_Version9()                       
    model = ai_settings['model']
    
    #checkpoint_filepath = 'ModelCheckpoint/CNN_Attention_BiLSTM_Version3/Checkpoint'

    #early_stopping = EarlyStopping(monitor='val_loss', patience=500)  # patience 번이상 더 좋은 결과가 없으면 학습을 멈춤
    #callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
    #                                          monitor='val_loss', 
    #                                          save_best_only=True, verbose=1)
    
    #ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, verbose=1, mode='min',monitor='val_loss')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                           patience=5, min_lr=0.001, mode='min',verbose=1)
    #wandb.init(project="simulation_buy", entity="SeongJae-Yoo")
    #wandb.run.name = 'Bi_LSTM_layers_4'
    
    #wandb.run.name = 'test_222'
    # Save a model file manually from the current directory:
    #wandb.save('model-best.h5')

    # restore the model file "model.h5" from a specific run by user "lavanyashukla"
# in project "save_and_restore" from run "10pr4joa"
    #best_model = wandb.restore('model-best.h5', run_path="SeongJae-Yoo/samyangfoods/runs/3db1mv6m")


# use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
    #model.load_weights(best_model.name)
    #model.load_weights('model-best.h5')
    # generted run ID로 하고 싶다면 다음과 같이 쓴다.
    # wandb.run.name = wandb.run.id
    #wandb.run.save()
    # wandb.log({"gradients": wandb.Histogram(numpy_array_or_sequence)})
    # wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})

    # wandb   Hyperparameter Sweeps   시각화 할 수 있는 자료 (아래 확인)
    #https://colab.research.google.com/drive/1gKixa6hNUB8qrn1CfHirOfTEQm0qLCSS#scrollTo=1gD9qhA9yOYs
    
    #wandb_callback = WandbCallback(monitor='val_loss',save_model=True,mode='min',log_weights=True,log_evaluation=True,validation_steps=5,verbose=1)
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.0001, mode='min',verbose=1)
    
    model.fit(shuffled_data["X_train"], shuffled_data["y_train"],
                        batch_size=ai_settings['batch_size'],
                        epochs=ai_settings['epochs'],
                        validation_data=(shuffled_data["X_test"], shuffled_data["y_test"]),
                        callbacks=[reduce_lr],
                        verbose=1)

    scaled_data = load_data(df=dataset.copy(), n_steps=ai_settings['n_steps'], test_size=ai_settings['test_size'],
                            shuffle=False)

    # result = evaluate(scaled_data, model)
    # print(f"result: {result}")

    
    # train_huber_loss, train_mae, train_rmse,test_huber_loss, test_mae, test_rmse = evaluate(scaled_data, model)  
    # print(f"train_huber_loss, train_mae, train_rmse: {train_huber_loss,train_mae, train_rmse}")      
    # print(f"test_huber_loss, test_mae, test_rmse: {test_huber_loss,test_mae, test_rmse}")      

    # mse = evaluate(scaled_data, model)
    # print(f"Mean Squared Error: {mse}")



    # 예측 가격
    future_price = predict(scaled_data, model, n_steps=ai_settings['n_steps'])

    # 스케일링 된 예측 결과
    scaled_y_pred = model.predict(scaled_data['X_test'])
    # 실제 값으로 변환 된 결과
    y_pred = np.squeeze(scaled_data['column_scaler']['close'].inverse_transform(scaled_y_pred))

    if ai_settings['is_used_predicted_close']:
        close = y_pred[-1] # 예측 그래프에서의 종가
    else:
        close = dataset.iloc[-1]['close'] # 실제 종가

    # ratio : 예상 상승률
    ratio = (future_price - close) / close * 100

    msg = f"After {ai_settings['lookup_step']}: {int(close)} -> {int(future_price)}"

    if ratio > 0: # lookup_step(분, 일) 후 상승 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯅ '
    elif ratio < 0: # lookup_step(분, 일) 후 하락 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯆ '
    print(msg, end=' ')
    return ai_settings['ratio_cut'] >= ratio # ratio_cut(목표 수익률) 보다 ratio가 작으면 True 반환(필터링 대상)



def create_training_engine(db_name):
    return pymysql.connect(
        host=cf.db_ip,
        port=int(cf.db_port),
        user=cf.db_id,
        password=cf.db_passwd,
        db=db_name,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

#
def ai_filter(ai_filter_num, engine, until=datetime.datetime.today()):
        if ai_filter_num == 1:
            ai_settings = {
                        "n_steps": 100, # 시퀀스 데이터를 몇개씩 담을지 설정
                        "lookup_step": 30, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                        "test_size": 0.2, # train 범위 : test_size 가 0.2 이면 X_train, y_train에 80% 데이터로 트레이닝 하고 X_test,y_test에 나머지 20%로 테스트를 하겠다는 의미
                        "n_layers": 4, # LSTM layer 개수
                        "units": 4, # LSTM neurons 개수
                        "dropout": 0.2, # overfitting 방지를 위해 몇개의 노드를 죽이고 남은 노드들을 통해서만 훈련을 하는 것(0.2 -> 20%를 죽인다)
                        "loss": "mae", # loss : 최적화 과정에서 최소화될 손실 함수(loss function)를 설정 # mae : mean absolute error (평균 절대 오차)
                        "optimizer": "adam", # optimizer : 최적화 알고리즘 선택
                        "batch_size": 164, # 각 학습 반복에 사용할 데이터 샘플 수
                        "epochs": 2, # 몇 번 테스트 할지
                        "ratio_cut": 3, #단위:(%) lookup_step 기간 뒤 ratio_cut(%) 만큼 증가 할 것이 예측 된다면 매수
                        "table": "daily_craw",  #분석 시 daily_craw(일별데이터)를 이용 할지 min_craw(분별데이터)를 이용 할지 선택. ** 주의: min_craw 선택 시 최근 1년 데이터만 있기 때문에 simulator_func_mysql.py에서 self.simul_start_date를 최근 1년 전으로 설정 필요
                        "is_used_predicted_close" : True # ratio(예상 상승률) 계산 시 예측 그래프의 close 값을 이용 할 경우 True, 실제 close 값을 이용할 시 False
                    }

            tr_engine = create_training_engine(ai_settings['table'])

        elif ai_filter_num == 2:
            ai_settings = {
                        "n_steps": 100, # 시퀀스 데이터를 몇개씩 담을지 설정
                        "lookup_step": 390, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                        "test_size": 0.3, # train 범위 : test_size 가 0.2 이면 X_train, y_train에 80% 데이터로 트레이닝 하고 X_test,y_test에 나머지 20%로 테스트를 하겠다는 의미
                        "n_layers": 4, # layer 개수
                        "units": 4, # neurons 개수
                        "dropout": 0.5, # overfitting 방지를 위해 몇개의 노드를 죽이고 남은 노드들을 통해서만 훈련을 하는 것(0.2 -> 20%를 죽인다)
                        "loss": "mae", # loss : 최적화 과정에서 최소화될 손실 함수(loss function)를 설정
                        "optimizer": "RMSprop", # optimizer : 최적화 알고리즘 선택
                        "batch_size": 164, # 각 학습 반복에 사용할 데이터 샘플 수
                        "epochs": 10, # 몇 번 테스트 할지
                        "ratio_cut": 3, #단위:(%) lookup_step 기간 뒤 ratio_cut(%) 만큼 증가 할 것이 예측 된다면 매수
                        "table": "min_craw",  #분석 시 daily_craw(일별데이터)를 이용 할지 min_craw(분별데이터)를 이용 할지 선택. ** 주의: min_craw 선택 시 최근 1년 데이터만 있기 때문에 simulator_func_mysql.py에서 self.simul_start_date를 최근 1년 전으로 설정 필요
                        "is_used_predicted_close" : False # ratio(예상 상승률) 계산 시 예측 그래프의 close 값을 이용 할 경우 True, 실제 close 값을 이용할 시 False
                    }

            tr_engine = create_training_engine(ai_settings['table'])
    # 2022-10-31 Written by SEONGJAE-YOO (Commits on Oct 31, 2022)
    #### 함수로 모델 사용 !@ 
    # model 함수 부분만 바꾸어서 다른 모델 실험할 수 있습니다.
    #maxlen=5, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos')
    #CNN_Attention_BiLSTM_Version3(maxlen=5, units=21, dropout=0.3, n_steps=1, LOSS = "mae", optimizer= 'cos'):
        elif ai_filter_num == 3:
            ai_settings = {   
                        "model" : None,
                        "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                        "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                        "test_size": 0.3,
                        "batch_size": 32,
                        "epochs": 100,
                        "ratio_cut": 1,
                        "table": "daily_craw",
                        "is_used_predicted_close" : True #false는 단한종목도 사지 않는다.
                    }

            tr_engine = create_training_engine(ai_settings['table'])



            # DISTINCT : 중복된 컬럼 제거
            try:   
                buy_list = engine.execute("""
                    SELECT DISTINCT code_name FROM realtime_daily_buy_list
                """).fetchall()
            except (InternalError, ProgrammingError) as err:
                if 'Table' in str(err):
                    print(f"{err} \n realtime_daily_buy_list 테이블이 존재 하지 않습니다. \n 콜렉터를 실행해주세요 ")
                else:
                    print(f"{err} \n 데이터베이스가 존재 하지 않습니다. \n 콜렉터를 실행해주세요 ")
                exit(1)
                 
            feature_columns = ["close", "volume", "open", "high", "low"]
            # feature_columns   = [ 'close', 'open', 'high', 'low',
            #         'volume', 'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
            #         'clo100', 'clo120','yes_clo5', 'yes_clo10', 'yes_clo20', 'yes_clo40', 'yes_clo60','yes_clo80','yes_clo100', 'yes_clo120'
            #         ] 
            filtered_list = []
            for code_name, in buy_list:
                print(f"{code_name} 종목 분석 중....")

                sql = """
                    SELECT {} FROM `{}`
                    WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, until)
                # pandas(pd) read_sql 을 사용하면 sql, engine을 넘겼을 때 return 값을 바로 데이터프레임으로 받을 수 있음
                df = pd.read_sql(sql, tr_engine)

                # 데이터가 1개(1일 or 1분)가 넘지 않으면 예측도가 떨어지기 때문에 필터링
                if len(df) < 1:
                    filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어서 realtime_daily_buy_list 에서 제외")
                    continue
                try:
                    if 1<= len(df) <=5000:
                        ai_settings['model'] = CNN_Attention_BiLSTM_Version27()  # 셀트리온헬스케어 model-best.h5 -> pretty-sweep-110 사용
                        filtered = filtered_by_basic_lstm(df, ai_settings)
                    elif len(df) >5000:       
                        ai_settings['model'] = CNN_Attention_BiLSTM_Version27()
                        filtered = filtered_by_basic_lstm_v2(df, ai_settings)       
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어서 realtime_daily_buy_list 에서 제외")
                    filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 filtered_list(필터링 종목)에 해당 종목을 append
                if filtered:
                    print(f"기준에 부합하지 않으므로 realtime_daily_buy_list 에서 제외")
                    filtered_list.append(code_name)
                    
        elif ai_filter_num == 4:
            ai_settings = {   
                        "model" : None,
                        "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                        "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                        "test_size": 0.3,
                        "batch_size": 32,
                        "epochs": 100,
                        "ratio_cut": 0,
                        "table": "daily_craw",
                        "is_used_predicted_close" : True #false는 단한종목도 사지 않는다.
                    }

            tr_engine = create_training_engine(ai_settings['table'])



            # DISTINCT : 중복된 컬럼 제거
            try:   
                buy_list = engine.execute("""
                    SELECT DISTINCT code_name FROM realtime_daily_buy_list
                """).fetchall()
            except (InternalError, ProgrammingError) as err:
                if 'Table' in str(err):
                    print(f"{err} \n realtime_daily_buy_list 테이블이 존재 하지 않습니다. \n 콜렉터를 실행해주세요 ")
                else:
                    print(f"{err} \n 데이터베이스가 존재 하지 않습니다. \n 콜렉터를 실행해주세요 ")
                exit(1)
                 
            feature_columns = ["close", "volume", "open", "high", "low"]
            # feature_columns   = [ 'close', 'open', 'high', 'low',
            #         'volume', 'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
            #         'clo100', 'clo120','yes_clo5', 'yes_clo10', 'yes_clo20', 'yes_clo40', 'yes_clo60','yes_clo80','yes_clo100', 'yes_clo120'
            #         ] 
            filtered_list = []
            for code_name, in buy_list:
                print(f"{code_name} 종목 분석 중....")

                sql = """
                    SELECT {} FROM `{}`
                    WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, until)
                # pandas(pd) read_sql 을 사용하면 sql, engine을 넘겼을 때 return 값을 바로 데이터프레임으로 받을 수 있음
                df = pd.read_sql(sql, tr_engine)

                # 데이터가 1개(1일 or 1분)가 넘지 않으면 예측도가 떨어지기 때문에 필터링
                if len(df) < 1:
                    filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어서 realtime_daily_buy_list 에서 제외")
                    continue
                try:
                    if 1<= len(df) <=5000:
                        ai_settings['model'] = CNN_Attention_BiLSTM_Version27()  # 셀트리온헬스케어 model-best.h5 -> pretty-sweep-110 사용
                        filtered = filtered_by_basic_lstm(df, ai_settings)
                    elif len(df) >5000:       
                        ai_settings['model'] = CNN_Attention_BiLSTM_Version27()
                        filtered = filtered_by_basic_lstm_v2(df, ai_settings)       
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어서 realtime_daily_buy_list 에서 제외")
                    filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 filtered_list(필터링 종목)에 해당 종목을 append
                if filtered:
                    print(f"기준에 부합하지 않으므로 realtime_daily_buy_list 에서 제외")
                    filtered_list.append(code_name)

        # filtered_list에 있는 종목들을 realtime_daily_buy_list(매수리스트)에서 제거
        # 모든 조건문에서 filtered_list를 생성해줘야 함
        if len(filtered_list) > 0:
            engine.execute(f"""
                DELETE FROM realtime_daily_buy_list WHERE code_name in ({','.join(map('"{}"'.format, filtered_list))})
            """)


if __name__ == '__main__':
    # 모의투자, 실전투자 일때만 들어오는 함수
    filter_by_ai(*sys.argv[1:])
