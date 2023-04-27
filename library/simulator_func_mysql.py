ver = "#version 1.3.10"
print(f"simulator_func_mysql Version: {ver}")
import sys
is_64bits = sys.maxsize > 2**32
if is_64bits:
    print('64bit 환경입니다.')
else:
    print('32bit 환경입니다.')

# SQLAlchemy is the Python SQL toolkit and Object Relational Mapper
from sqlalchemy import event

import pymysql.cursors

from library.daily_crawler import *
from library.logging_pack import *
from library import cf
from pandas import DataFrame
import re
import datetime
from sqlalchemy import create_engine

# 볼린저밴드 알고리즘을 위해 추가된 부분 
from library.trading_algorithms import BBands
import numpy as np

#sell_ai function
from ai.SPPModel import CNN_Attention_BiLSTM_Version11 ,load_data,predict,DataNotEnough, CNN_Attention_BiLSTM_Version17, CNN_Attention_BiLSTM_Version9, CNN_Attention_BiLSTM_Version27,BiGRU_CNN_BiLSTM_Attention_version2, BiLSTM_Attention_CNN_version2, CNN_BiLSTM_Attention_version2 
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf  
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

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


pymysql.install_as_MySQLdb()


class simulator_func_mysql:
    def __init__(self, simul_num, op, db_name):
        self.simul_num = int(simul_num)

        # scraper할 때 start date 가져오기 위해서
        if self.simul_num == -1:
            self.date_setting()

        # option이 reset일 경우 실행
        elif op == 'reset':
            self.op = 'reset'
            self.simul_reset = True
            self.db_name = db_name
            self.variable_setting()
            self.rotate_date()

        # option이 real일 경우 실행(시뮬레이터와 무관)
        # 모의투자 할때 사용됨
        elif op == 'real':
            self.op = 'real'
            self.simul_reset = False
            self.db_name = db_name
            self.variable_setting()

        #  option이 continue 일 경우 실행
        elif op == 'continue':
            self.op = 'continue'
            self.simul_reset = False
            self.db_name = db_name
            self.variable_setting()
            self.rotate_date()
        else:
            print("simul_num or op 어느 것도 만족 하지 못함 simul_num : %s ,op : %s !!", simul_num, op)

    # 마지막으로 구동했던 시뮬레이터의 날짜를 가져온다.
    def get_jango_data_last_date(self):
        sql = "SELECT date from jango_data order by date desc limit 1" # limit : 출력 행 갯수 제한하는 구문 , limit 1 은 한행만 출력해준다.(desc가 내림차순 순서이므로 가장 최근의 날짜가 출력된다)
        return self.engine_simulator.execute(sql).fetchall()[0][0]

    # 모든 테이블을 삭제 하는 함수
    def delete_table_data(self):
        logger.info('delete_table_data !!!!')
        if self.is_simul_table_exist(self.db_name, "all_item_db"):
            sql = "drop table all_item_db"
            self.engine_simulator.execute(sql)
            # 만약 jango data 컬럼을 수정하게 되면 테이블을 삭제하고 다시 생성이 자동으로 되는데 이때 삭제했으면 delete가 안먹힌다. 그래서 확인 후 delete

        if self.is_simul_table_exist(self.db_name, "jango_data"):
            sql = "drop table jango_data"
            self.engine_simulator.execute(sql)

        if self.is_simul_table_exist(self.db_name, "realtime_daily_buy_list"):
            sql = "drop table realtime_daily_buy_list"
            self.engine_simulator.execute(sql)

    # realtime_daily_buy_list 테이블의 check_item컬럼에 특정 종목의 매수 시간을 넣는 함수
    def update_realtime_daily_buy_list(self, code, min_date):
        sql = "update realtime_daily_buy_list set check_item = '%s' where code = '%s'"
        self.engine_simulator.execute(sql % (min_date, code))

    # 시뮬레이션 옵션 설정 함수
    def variable_setting(self):
        # 아래 if문으로 들어가기 전까지의 변수들은 모든 알고리즘에 공통적으로 적용 되는 설정
        # 오늘 날짜를 설정
        self.date_setting()
        # 시뮬레이팅이 끝나는 날짜.
        self.simul_end_date = self.today
        self.start_min = "0900"

        # 아래 3개는 분별시뮬레이션 옵션
        # (use_min, only_nine_buy 변수만 각각의 알고리즘에 붙여 넣기 해서 사용)
        # 분별 시뮬레이션을 사용하고 싶을 경우 아래 옵션을 True로 변경하여 사용
        # 일별 시뮬레이션을 사용하고 싶을 경우 False 
        self.use_min = False
        # 아침 9시에만 매수를 하고 싶은 경우 True, 9시가 아니어도 매수를 하고 싶은 경우 False(분별 시뮬레이션 적용 가능 / 일별 시뮬레이션은 9시에만 매수, 매도)
        self.only_nine_buy = True
        # self.buy_stop옵션은 수정 필요가 없음. self.only_nine_buy 옵션을 True로 하게 되면 시뮬레이터가 9시에 매수 후에 self.buy_stop을 true로 변경해서 당일에는 더이상 매수하지 않도록 설정함
        self.buy_stop = False

        # AI알고리즘 사용 여부 (기본값 False 설정)
        self.use_ai = False  # ai 알고리즘 사용 시 True 사용 안하면 False
        self.ai_filter_num = 1  # ai 알고리즘 선택

        # 실시간 조건 매수 옵션 (기본값 False 설정)
        # self.only_nine_buy 옵션을 반드시 False로 설정해야함
        # self.use_min 옵션이 반드시 True로 설정이 되어야함
        # 실시간 조건 매수 알고리즘 선택 (1,2,3..)
        self.trade_check_num = False

        print("self.simul_num!!! ", self.simul_num)

        ###!@####################################################################################################################
        # 아래 부터는 알고리즘 별로 별도의 설정을 해주는 부분

        if self.simul_num in (1,4):
            # 시뮬레이팅 시작 일자(분 별 시뮬레이션의 경우 최근 1년 치 데이터만 있기 때문에 start_date 조정 필요)
            self.simul_start_date = "19000802"

            ######### 알고리즘 선택 #############
            # 매수 리스트 설정 알고리즘 번호
            self.db_to_realtime_daily_buy_list_num = 1

            # 매도 리스트 설정 알고리즘 번호
            self.sell_list_num = 2
            ###################################

            # 초기 투자자금(시뮬레이션에서의 초기 투자 금액. 모의투자는 신청 당시의 금액이 초기 투자 금액이라고 보시면 됩니다)
            # 주의! start_invest_price 는 모의투자 초기 자본금과 별개. 시뮬레이션에서만 적용.
            # 키움증권 모의투자의 경우 초기에 모의투자 신청 할 때 설정 한 금액으로 자본금이 설정됨
            self.start_invest_price = 10000000

            # ex. 10만원 씩 분산 투자 해서 설정한 경우 / start_invest_price 에서 invest_unit 변수 값 만큼 매일 분산해서 여러종목을 투자한다.
            # 매일 한종목에 invest_unit 변수 값만큼 투자 하고 나머지 금액은 여러 종목마다  invest_unit 값만큼 투자한다. (분산 투자 개념)
            self.invest_unit = 10000000

            # 자산 중 최소로 남겨 둘 금액
            self.limit_money = 0

            # 익절 수익률 기준치
            self.sell_point = 10

            # 손절 수익률 기준치
            self.losscut_point = -5

            self.invest_limit_rate = 1.02
            self.invest_min_limit_rate = 0.97       
  
            
            
            # 분별 시뮬레이션을 사용하고 싶을 경우 (simul_num을 4로 입력)
            if self.simul_num ==4:
                self.simul_start_date = '20210902'
                self.use_min = True
                self.only_nine_buy = False
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택

        elif self.simul_num == 2:
            # 시뮬레이팅 시작 일자
            self.simul_start_date = "20220802"

            ######### 알고리즘 선택 #############
            # 매수 리스트 설정 알고리즘 번호 # 5 / 20 이동 평균선 골든크로스 buy
            self.db_to_realtime_daily_buy_list_num = 1
            # 매도 리스트 설정 알고리즘 번호 # 5 / 20 이동 평균선 
            self.sell_list_num = 2
            ###################################
            # 초기 투자자금
            # 주의! start_invest_price 는 모의투자 초기 자본금과 별개. 시뮬레이션에서만 적용.
            # 키움증권 모의투자의 경우 초기에 모의투자 신청 할 때 설정 한 금액으로 자본금이 설정됨
            self.start_invest_price = 9448076
            # 매수 금액
            self.invest_unit = 1000000

            # 자산 중 최소로 남겨 둘 금액
            self.limit_money = 0
            # # 익절 수익률 기준치
            self.sell_point = 5
            # 손절 수익률 기준치
            self.losscut_point = -5
            
            self.invest_limit_rate = 1.02
           
            self.invest_min_limit_rate = 0.97


        elif self.simul_num == 3:

            # 시뮬레이팅 시작 일자

            self.simul_start_date = "20220802"

            ######### 알고리즘 선택 #############

            # 매수 리스트 설정 알고리즘 번호
            # 5 / 40 골든크로스 buy
            self.db_to_realtime_daily_buy_list_num = 2

            # 매도 리스트 설정 알고리즘 번호
            # 5 / 40 이동 평균선 데드크로스 
            self.sell_list_num = 3

            ###################################

            # 초기 투자자금
            # 주의! start_invest_price 는 모의투자 초기 자본금과 별개. 시뮬레이션에서만 적용.
            # 키움증권 모의투자의 경우 초기에 모의투자 신청 할 때 설정 한 금액으로 자본금이 설정됨
            self.start_invest_price = 9448076

            # 매수 금액
            self.invest_unit = 100000

            # 자산 중 최소로 남겨 둘 금액
            self.limit_money = 0

            # 익절 수익률 기준치
            self.sell_point = 10

            # 손절 수익률 기준치
            self.losscut_point = -5

            # 실전/모의 봇 돌릴 때 매수하는 순간 종목의 최신 종가 보다 2% 이상 오른 경우 사지 않도록 하는 설정(변경 가능)
            self.invest_limit_rate = 1.02
            # 실전/모의 봇 돌릴 때 매수하는 순간 종목의 최신 종가 보다 -3% 이하로 떨어진 경우 사지 않도록 하는 설정(변경 가능)
            self.invest_min_limit_rate = 0.97
              

        elif self.simul_num in (5,6):    
            self.simul_start_date = "20220802"

            ######### 알고리즘 선택 #############
            # 매수 리스트 설정 알고리즘 번호
            self.db_to_realtime_daily_buy_list_num = 5

            self.interval_month = 3

            # 매도 리스트 설정 알고리즘 번호
            self.sell_list_num = 2
            ###################################


            self.start_invest_price = 9448076

            # 매수 금액
            self.invest_unit = 100000

            # 자산 중 최소로 남겨 둘 금액
            self.limit_money = 0

            # 익절 수익률 기준치
            self.sell_point = 10

            # 손절 수익률 기준치
            self.losscut_point = -5

            self.invest_limit_rate = 1.02
            self.invest_min_limit_rate = 0.97   

            # 관리, 불성실, 주의, 경고, 위험 제외 하고 buy 시뮬레이션을 사용하고 싶을 경우
            # 2022-09-30 Written by SEONGJAE-YOO
            if self.simul_num == 6:

                self.simul_start_date = '20220802'
                
                # 매도 리스트 설정 알고리즘 번호
                self.sell_list_num = 2
            
                # 익절 수익률 기준치
                self.sell_point = 5

                # 손절 수익률 기준치
                self.losscut_point = -1

                #분별 시뮬레이션을 사용하고 싶을 경우
                # self.use_min = True
                # self.only_nine_buy = False

        # 5 / 20 골든크로스 적용되고
        # 관리, 불성실, 주의, 경고, 위험 제외 하고 
        # volume * close (총 거래대금 금액)이 self.total_transaction_price( 변경가능) 원 보다 큰 주식을 buy 
        # 20일 평균거래량(거래량20이평선)대비 3배이상 거래량 터졌을때 강세( vol20 * '%s' < volume )이므로 buy
        # 전날보다  self.d1_diff 변수 값 (변경가능) 이상 올랐을 때 buy 
        
        # 2022-10-04 Written by SEONGJAE-YOO  (Commits on october 4, 2022)
        elif self.simul_num == 7:    
                    self.simul_start_date = "20220802"

                    ######### 알고리즘 선택 #############
                    # 매수 리스트 설정 알고리즘 번호
                    self.db_to_realtime_daily_buy_list_num = 6

                    self.interval_month = 3

                    # 5 / 40 moving average Death Cross sell list setting 알고리즘 번호
                    self.sell_list_num = 3
                    ###################################


                    self.start_invest_price = 9448076

                    # 매수 금액
                    self.invest_unit = 100000

                    # 자산 중 최소로 남겨 둘 금액
                    self.limit_money = 0

                    # 익절 수익률 기준치
                    self.sell_point = 3

                    # 손절 수익률 기준치
                    self.losscut_point = -3

                    self.invest_limit_rate = 1.01
                    self.invest_min_limit_rate = 0.99   
                    # volume * close (총 거래대금 금액) 의 변수: total_transaction_price
                    self.total_transaction_price = 10000000

                    ## 분별 시뮬레이션을 사용하고 싶을 경우()
                    # self.use_min = True
                    # self.only_nine_buy = False

                    self.vol_mul = 3 
                    self.d1_diff = 2 

       #  상대 모멘텀 / 절대 모멘텀 (Relative Strength Momentum function, Absolute Momentum function)
       # 2022-10-04 Written by SEONGJAE-YOO  (Commits on october 4, 2022)
        elif self.simul_num in (8,9,10,11):
                    # 매수 리스트 설정 알고리즘 번호(Absolute Momentum code ver)
                    self.db_to_realtime_daily_buy_list_num = 7
                    # 매도 리스트 설정 알고리즘 번호(Absolute Momentum code ver)
                    self.sell_list_num = 5
                    # 시뮬레이팅 시작 일자(분 별 시뮬레이션의 경우 최근 1년 치 데이터만 있기 때문에 start_date 조정 필요)
                    self.simul_start_date = "20220802"
                    # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 250일->1년)
                    self.day_before = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                    # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                    self.diff_point = 10 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                  
                    self.invest_unit = 100000
                    self.start_invest_price = 9448076

                    # 자산 중 최소로 남겨 둘 금액
                    self.limit_money = 0

                    # 익절 수익률 기준치
                    self.sell_point = 3
                    # 손절 수익률 기준치
                    self.losscut_point = -2
                    # 실전/모의 봇 돌릴 때 매수하는 순간 종목의 최신 종가 보다 1% 이상 오른 경우 사지 않도록 하는 설정(변경 가능)
                    self.invest_limit_rate = 1.01
                    # 실전/모의 봇 돌릴 때 매수하는 순간 종목의 최신 종가 보다 -2% 이하로 떨어진 경우 사지 않도록 하는 설정(변경 가능)
                    self.invest_min_limit_rate = 0.98   

            
                    if self.simul_num == 9:
                        # 매수 리스트 설정 알고리즘 번호 (Absolute Momentum query ver)
                        self.db_to_realtime_daily_buy_list_num = 8
                        # 매도 리스트 설정 알고리즘 번호 (Absolute Momentum query ver)
                        self.sell_list_num = 6
            
                    elif self.simul_num == 10 :
                        # 매수 리스트 설정 알고리즘 번호 (Absolute Momentum query ver)
                        self.db_to_realtime_daily_buy_list_num = 8
                        # 매도 리스트 설정 알고리즘 번호 (Absolute Momentum query ver + losscut point 추가)
                        self.sell_list_num = 7
                        # 손절 수익률 기준치
                        self.losscut_point = -2
            
                    elif self.simul_num == 11:
                        # 매수 리스트 설정 알고리즘 번호 (Relative Strength Momentum query ver)
                        self.db_to_realtime_daily_buy_list_num = 9

                        # 매도 리스트 설정 알고리즘 번호 (Absolute Momentum query ver + losscut point 추가)
                        self.sell_list_num = 7
   
                        
                          # # 분별 시뮬레이션 옵션
                        self.use_min = True
                        self.only_nine_buy = False  

        # 2022-10-08 Written by SEONGJAE-YOO (Commits on Oct 8, 2022)
        # 실시간 조건 매수 (realtime_daily_buy_list 데이터 에서 trade_check_num 알고리즘에 따라 매수하는 전략)
        # self.only_nine_buy 옵션을 반드시 False로 설정해야함 (실시간 조건 매수 조건)
        # self.use_min 옵션이 반드시 True로 설정이 되어야함 (실시간 조건 매수 조건)
        # 결론 - 분별 시뮬레이션 할때만 실시간 조건 매수를 할 수 있습니다. !@
        elif self.simul_num in (12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32):
            
            self.simul_start_date = "20220502"

            ######### 알고리즘 선택 #############
            # 매수 리스트 설정 알고리즘 번호
            self.db_to_realtime_daily_buy_list_num = 10

            # 매도 리스트 설정 알고리즘 번호
            self.sell_list_num = 8
            ###################################

            # 초기 투자자금(시뮬레이션에서의 초기 투자 금액. 모의투자는 신청 당시의 금액이 초기 투자 금액이라고 보시면 됩니다)
            # 주의! start_invest_price 는 모의투자 초기 자본금과 별개. 시뮬레이션에서만 적용.
            # 키움증권 모의투자의 경우 초기에 모의투자 신청 할 때 설정 한 금액으로 자본금이 설정됨
            self.start_invest_price = 10000000

            # 매수 금액  
            self.invest_unit = 2000000

            # 자산 중 최소로 남겨 둘 금액
            self.limit_money = 0

            # 익절 수익률 기준치
            self.sell_point = 10

            # 손절 수익률 기준치
            self.losscut_point = -2
            
            # 실전/모의 봇 돌릴 때 매수하는 순간 종목의 최신 종가 보다 2% 이상 오른 경우 사지 않도록 하는 설정(변경 가능)
            self.invest_limit_rate = 1.02
            # 실전/모의 봇 돌릴 때 매수하는 순간 종목의 최신 종가 보다 -3% 이하로 떨어진 경우 사지 않도록 하는 설정(변경 가능)
            self.invest_min_limit_rate = 0.97

            # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 250일->1년)
            self.day_before = 60 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
            # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
            self.diff_point = 10 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
            
            # volume * close (총 거래대금 금액) 의 변수: total_transaction_price
            self.total_transaction_price = 100000
            self.vol_mul = 2
            self.d1_diff = 2 
            self.interval_month = 12                       
            
            # # 일별 시뮬레이션을 사용하고 싶을 경우 False 
            # self.use_min = True
            # # 아침 9시에만 매수를 하고 싶은 경우 True, 9시가 아니어도 매수를 하고 싶은 경우 False(분별 시뮬레이션 적용 가능 / 일별 시뮬레이션은 9시에만 매수, 매도)
            # self.only_nine_buy = True
            

            self.volume_up = 2  # 특정 거래대금 보다 x배 이상 증가 할 경우 매수
            self.rarry_k = 0.6

            
            self.audit = str("정상")
            self.margin = 30
            self.remarks_manage = str("%관리종목%")
            self.remarks_stop = str("%거래정지%")
            self.stock_market_a = str("거래소")
            self.stock_market_b = str("코스닥")  #           info.stock_market IN ("거래소", "코스닥")
            self.stock_market_c = str("KONEX")
            self.stock_market_d = str("ETF")
            self.category0_a = str("우량기업")
            self.category0_b = str("외국기업")
            self.category0_c = str("중견기업")

            if self.simul_num == 13:
                
                self.trade_check_num = 2
                # 매수하는 순간 종목의 최신 종가 보다 1% 이상 오른 경우 사지 않도록 하는 설정(변경 가능)
                self.invest_limit_rate = 1.01
                # 매수하는 순간 종목의 최신 종가 보다 -2% 이하로 떨어진 경우 사지 않도록 하는 설정(변경 가능)
                self.invest_min_limit_rate = 0.98
               
            # 래리윌리엄스 변동성 돌파 전략
            # 2022-10-08 Written by SEONGJAE-YOO (Commits on Oct 8, 2022)
            elif self.simul_num == 14:
                self.trade_check_num = 3
             
            # 2022-10-17 Written by SEONGJAE-YOO (Commits on Oct 17, 2022)
            elif self.simul_num == 15:
                 # 매수 리스트 설정 알고리즘 번호 
                self.db_to_realtime_daily_buy_list_num = 12
                 # 매도 리스트 설정 알고리즘 번호
                self.sell_list_num = 7
                self.trade_check_num = 3
            
            elif self.simul_num == 16:
                 # 매수 리스트 설정 알고리즘 번호 
                self.db_to_realtime_daily_buy_list_num = 12
                 # 매도 리스트 설정 알고리즘 번호
                self.sell_list_num = 7
                self.trade_check_num = 3


                # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 1  # ai 알고리즘 선택

            # 볼린저밴드 알고리즘
            elif self.simul_num == 17:
                self.db_to_realtime_daily_buy_list_num = 14
                # 매도리스트
                self.sell_list_num = 9
                self.trade_check_num = 3

            elif self.simul_num == 18:       
                self.db_to_realtime_daily_buy_list_num = 15
                self.sell_list_num = 10
                self.trade_check_num = 3
                self.simul_start_date = "20210813"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 250일->1년)
                self.day_before = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                self.diff_point = 10 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.margin = 40
                # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택
                
            # 1달전 , 6달전 모멘텀 전략 + 볼린저밴드 전략
            # 2022-10-18 Written by SEONGJAE-YOO (Commits on Oct 18, 2022)
            elif self.simul_num == 19:       
                self.db_to_realtime_daily_buy_list_num = 16
                self.sell_list_num = 11
                self.trade_check_num = 3
                self.simul_start_date = "20210713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 250일->1년)
                self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                self.date_before_b = 120 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                self.diff_point = 10 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                
                self.margin = 40    


            
            # 매수 전략    
            #  1달전 , 6달전 모멘텀 전략 + 볼린저밴드 전략 + 래리 윌리엄스 변동성 돌파 매수 전략
            # 2022-10-18 Written by SEONGJAE-YOO (Commits on Oct 18, 2022)
            #ai 적용시 20211228 일 부터 realtime_daily_buy_list 테이블에 들어감
            elif self.simul_num == 20:        
                self.db_to_realtime_daily_buy_list_num = 16
                self.sell_list_num = 11
                self.trade_check_num = 3
                self.simul_start_date = "20211001"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 250일->1년)
                self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                self.date_before_b = 120 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                self.diff_point = 10 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                
                self.margin = 40  
                # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택
                  
        # 대략 1달전 ,3달전, 6달전 ,12달전  모멘텀 전략 + 볼린저밴드 전략 +래리 윌리엄스 변동성 돌파 매수 전략
        # 2022-10-18 Written by SEONGJAE-YOO (Commits on Oct 18, 2022)
            elif self.simul_num == 21:       

                self.db_to_realtime_daily_buy_list_num = 18
                # 20210713 기준
                #self.sell_list_num = 2 # 1744971
                #self.sell_list_num = 14 #202040
                #self.sell_list_num = 1 # -251539
                #self.sell_list_num = 9 #-257408
                #self.sell_list_num = 3 #151149
                #self.sell_list_num = 4 # 1571238
                #self.sell_list_num = 5 # 1392415
                #self.sell_list_num = 6 #1392415
                #self.sell_list_num = 7 #-257013
                #self.sell_list_num = 8 #-820100
                #self.sell_list_num = 10 # -488224
                #self.sell_list_num = 11 # 마이너스
                #self.sell_list_num = 12 # error
                #self.sell_list_num = 13 #error
                self.sell_list_num = 2
                #self.trade_check_num = 3
                self.simul_start_date = "20210713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                self.date_before_b = 60 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                self.date_before_c = 120
                self.date_before_d = 240

                self.diff_point = 10 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                
                self.margin = 40
        #!@
        # 1달전 ,3달전, 6달전 ,12달전  상대 모멘텀 전략(Relative Strength Momentum) + 볼린저밴드 전략 + 래리 윌리엄스 변동성 돌파 매수 전략
        # 2022-10-18 Written by SEONGJAE-YOO (Commits on Oct 18, 2022)1744971
        # 20210513일 기준 , self.diff_point = 15
        # self.sell_list_num = 5  -> -1341002  
        # self.sell_list_num = 2 -> 825594
        # 20210113일 기준 self.sell_list_num = 2-> 930471
        # 20210113일 기준 self.sell_list_num = 5-> -2711579
        #########################################################33
        # self.diff_point = 7, self.sell_list_num = 5 -> -4278731
        # self.diff_point = 7, self.sell_list_num = 2 -> -750378
        # 고수익 방법 23알고리즘보다 안정적인 수익률 기대 못함
            elif self.simul_num == 22:       

                self.db_to_realtime_daily_buy_list_num = 18
                
                self.sell_list_num = 2
                #self.trade_check_num = 3
                self.simul_start_date = "20210112"
                #self.simul_start_date = "20200713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                self.date_before_b = 60 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                self.date_before_c = 120
                self.date_before_d = 240

                self.diff_point = 7 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                
                self.margin = 40         
                # AI알고리즘 사용 여부 
                # self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                # self.ai_filter_num = 3  # ai 알고리즘 선택    
        
        # 2022-10-18 Written by SEONGJAE-YOO (Commits on Oct 18, 2022)
        # 볼린저 밴드 0 설정
        # 1~11 
        # sell_list=5 -> 743791
        # sell_list=2 -> 36183
        #########################
        # 볼린저 밴드 0.1 설정
        # 162771    
        #     
        ##################################################3
        # 볼린저 밴드 0.25 설정 
        # sell_list=5 -> 134681
        # ma_period = 75 이동평균 기간에 따른 수익률 비교?
        # 방법 
        #  
            elif self.simul_num == 23:       

                
                self.db_to_realtime_daily_buy_list_num = 23
                
                self.sell_list_num = 5
                #self.trade_check_num = 3
                self.simul_start_date = "20220112"
                #self.simul_start_date = "20200713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                # self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                # self.date_before_b = 59 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # self.date_before_c = 123  
                # self.date_before_d = 245
                # self.day_before = 20    
                self.diff_point = 5 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.sell_diff_point = 5
                self.margin = 100         
                # # AI알고리즘 사용 여부 
                # self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                # self.ai_filter_num = 3  # ai 알고리즘 선택                 
            # 안정적인 방법 
            # 02 -27 일 이방법으로 시뮬레이션
            elif self.simul_num == 24:       

                
                self.db_to_realtime_daily_buy_list_num = 21
                
                self.sell_list_num = 5
                #self.trade_check_num = 3
                self.simul_start_date = "20210113"
                #self.simul_start_date = "20200713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                # self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                # self.date_before_b = 59 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # self.date_before_c = 123  
                # self.date_before_d = 245
                # self.day_before = 20    
                self.diff_point = 5 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.sell_diff_point = 3
                self.margin = 100         
                # # AI알고리즘 사용 여부 
                # self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                # self.ai_filter_num = 3  # ai 알고리즘 선택               

            elif self.simul_num == 25:       

                
                self.db_to_realtime_daily_buy_list_num = 21
                
                self.sell_list_num = 16
                #self.trade_check_num = 3
                self.simul_start_date = "20210113"
                #self.simul_start_date = "20200713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                # self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                # self.date_before_b = 59 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # self.date_before_c = 123  
                # self.date_before_d = 245
                # self.day_before = 20    
                self.diff_point = 5 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.sell_diff_point = 3
                self.margin = 100         
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택 
                
            # 2-28일 시뮬레이션
            elif self.simul_num == 26:       
                   
                self.db_to_realtime_daily_buy_list_num = 24
                
                self.sell_list_num = 16
                #self.trade_check_num = 3
                self.simul_start_date = "20210113"
                #self.simul_start_date = "20200713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                # self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                # self.date_before_b = 59 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # self.date_before_c = 123  
                # self.date_before_d = 245
                # self.day_before = 20    
                self.diff_point = 5 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.sell_diff_point = 3
                self.margin = 100         
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택   
            # ai model   "ratio_cut": 1, 변경 , "ratio_cut": -1 변경후 실험             
            elif self.simul_num == 27:       
                    
                self.db_to_realtime_daily_buy_list_num = 24
                
                self.sell_list_num = 16
                #self.trade_check_num = 3
                self.simul_start_date = "20210113"
                #self.simul_start_date = "20200713"
                # n일 전 종가 데이터를 가져올지 설정 (ex. 20 -> 장이 열리는 날 기준 20일 이니까 기간으로 보면 약 한 달, 240일->1년)
                # self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                # self.date_before_b = 59 # 단위 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                # self.date_before_c = 123  
                # self.date_before_d = 245
                # self.day_before = 20    
                self.diff_point = 5 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.sell_diff_point = 3
                self.margin = 100         
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택    
            # !@    
              
            elif self.simul_num == 28:       
                 
                self.db_to_realtime_daily_buy_list_num = 24                
                self.sell_list_num = 16
                self.simul_start_date = "20210113"            
                self.diff_point = 1 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.sell_diff_point = 1
                self.margin = 100      
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 3  # ai 알고리즘 선택   
            # ratio_cut = 0 은 수익률 마이너스 , ratio_cut =1 , 매도 할때는 -1 로 실험, !@
            # 29부터 논문 시뮬레이션 ! 가장 중요!          
            elif self.simul_num == 29:       
                 
                self.db_to_realtime_daily_buy_list_num = 24                
                self.sell_list_num = 18
                self.simul_start_date = "20210113"            
                self.diff_point = 1 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.margin = 100      
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 4  # ai 알고리즘 선택   
            #  BiGRU CNN BiLSTM Attention 모델사용하여 시뮬레이션
            elif self.simul_num == 30:       
                 
                self.db_to_realtime_daily_buy_list_num = 24                
                self.sell_list_num = 19
                self.simul_start_date = "20210113"            
                self.diff_point = 1 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.margin = 100      
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 5  # ai 알고리즘 선택
             #  BiLSTM Attention CNN 모델사용하여 시뮬레이션
            elif self.simul_num == 31:       
                 
                self.db_to_realtime_daily_buy_list_num = 24                
                self.sell_list_num = 20
                self.simul_start_date = "20210113"            
                self.diff_point = 1 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.margin = 100      
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 6  # ai 알고리즘 선택                

            elif self.simul_num == 32:       
                 
                self.db_to_realtime_daily_buy_list_num = 24                
                self.sell_list_num = 21
                self.simul_start_date = "20210113"            
                self.diff_point = 1 # 단위 % (모멘텀에서 n일 전 대비 종가(현재가)가 몇 프로 증가 했을 때 매수)
                self.margin = 100      
                # # AI알고리즘 사용 여부 
                self.use_ai = True  # ai 알고리즘 사용 시 True 사용 안하면 False
                self.ai_filter_num = 7  # ai 알고리즘 선택                






        else:
            logger.error(f"입력 하신 {self.simul_num}번 알고리즘에 대한 설정이 없습니다. simulator_func_mysql.py 파일의 variable_setting함수에 알고리즘을 설정해주세요. ")
            sys.exit(1)

        #########################################################################################################################
        self.db_name_setting()

        if self.op != 'real':
            # database, table 초기화 함수
            self.table_setting()

            # 시뮬레이팅 할 날짜를 가져 오는 함수
            self.get_date_for_simul()

            # 매도를 한 종목들 대상 수익
            self.total_valuation_profit = 0

            # 실제 수익 : 매도를 한 종목들 대상 수익 + 현재 보유 중인 종목들의 수익
            self.sum_valuation_profit = 0

            # 전재산 : 투자금액 + 실제 수익(self.sum_valuation_profit)
            self.total_invest_price = self.start_invest_price

            # 현재 총 투자한 금액
            self.total_purchase_price = 0

            # 현재 투자 가능한 금액(예수금) = (초기자본 + 매도한 종목의 수익) - 현재 총 투자 금액
            self.d2_deposit = self.start_invest_price

            # 일별 정산 함수
            self.check_balance()

            # 매수할때 수수료 한번, 매도할때 전체금액에 세금, 수수료
            self.tax_rate = 0.0025   
            self.fees_rate = 0.00015

            # 시뮬레이터를 멈춘 지점 부터 다시 돌리기 위해 사용하는 변수(중요X)!@
            self.simul_reset_lock = False

    # 데이터베이스와 테이블을 세팅하기 위한 함수
    def table_setting(self):
        print("self.simul_reset" + str(self.simul_reset))
        # 시뮬레이터를 초기화 하고 처음부터 구축하기 위한 로직
        if self.simul_reset:
            print("table reset setting !!! ")
            self.init_database() #데이터베이스 초기화 함수 실행
        # 시뮬레이터를 초기화 하지 않고 마지막으로 끝난 시점 부터 구동하기 위한 로직
        else:
            # self.simul_reset 이 False이고, 시뮬레이터 데이터베이스와, all_item_db 테이블, jango_table이 존재하는 경우 이어서 시뮬레이터 시작
            if self.is_simul_database_exist() and self.is_simul_table_exist(self.db_name,
                                                                            "all_item_db") and self.is_simul_table_exist(
                self.db_name, "jango_data"):
                self.init_df_jango()
                self.init_df_all_item()
                # 마지막으로 구동했던 시뮬레이터의 날짜를 가져온다.
                self.last_simul_date = self.get_jango_data_last_date()
                print("self.last_simul_date: " + str(self.last_simul_date))
            #    초반에 reset 으로 돌다가 멈춰버린 경우 다시 init 해줘야함
            else:
                print("초반에 reset 으로 돌다가 멈춰버린 경우 다시 init 해줘야함 ! ")
                self.init_database()
                self.simul_reset = True

    # 데이터베이스 초기화 함수
    def init_database(self):
        self.drop_database()
        self.create_database()
        self.init_df_jango()
        self.init_df_all_item()

    # 데이터베이스를 생성하는 함수
    def create_database(self):
        if self.is_simul_database_exist() == False:
            sql = 'CREATE DATABASE %s'
            self.db_conn.cursor().execute(sql % (self.db_name))
            self.db_conn.commit()

    # 데이터베이스를 삭제하는 함수
    def drop_database(self):
        if self.is_simul_database_exist():
            print("drop!!!!")
            sql = "drop DATABASE %s"
            self.db_conn.cursor().execute(sql % (self.db_name))
            self.db_conn.commit()

    # 데이터베이스의 존재 유무를 파악하는 함수.
    def is_simul_database_exist(self):
        sql = "SELECT 1 FROM Information_schema.SCHEMATA WHERE SCHEMA_NAME = '%s'"
        rows = self.engine_daily_buy_list.execute(sql % (self.db_name)).fetchall()
        print("rows : ", rows)
        if len(rows):
            return True
        else:
            return False

    # 오늘 날짜를 설정하는 함수
    def date_setting(self):
        self.today = datetime.datetime.today().strftime("%Y%m%d")
        self.today_detail = datetime.datetime.today().strftime("%Y%m%d%H%M")
        self.today_date_form = datetime.datetime.strptime(self.today, "%Y%m%d").date()

    # DB 이름 세팅 함수
    def db_name_setting(self):
        self.engine_simulator = create_engine(
            "mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" + cf.db_port + "/" + str(
                self.db_name),
            encoding='utf-8')
        if self.op != "real":
            # db_name을 setting 한다.
            self.db_name = "simulator" + str(self.simul_num)
            self.engine_simulator = create_engine(
                "mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" + cf.db_port + "/" + str(
                    self.db_name), encoding='utf-8')

        self.engine_daily_craw = create_engine(
            "mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" + cf.db_port + "/daily_craw",
            encoding='utf-8')

        self.engine_craw = create_engine(
            "mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" + cf.db_port + "/min_craw",
            encoding='utf-8')
        self.engine_daily_buy_list = create_engine(
            "mysql+mysqldb://" + cf.db_id + ":" + cf.db_passwd + "@" + cf.db_ip + ":" + cf.db_port + "/daily_buy_list",
            encoding='utf-8')

        # event.listen(self.engine_simulator, 'before_execute', escape_percentage, retval=True) # SQLAlchemy 라이브러리에서 가져온 event 함수
        # event.listen(self.engine_daily_craw, 'before_execute', escape_percentage, retval=True)
        # event.listen(self.engine_craw, 'before_execute', escape_percentage, retval=True)
        # event.listen(self.engine_daily_buy_list, 'before_execute', escape_percentage, retval=True)
        from library.open_api import escape_percentage
        event.listen(self.engine_simulator, 'before_execute', escape_percentage, retval=True)
        event.listen(self.engine_daily_craw, 'before_execute', escape_percentage, retval=True)
        event.listen(self.engine_craw, 'before_execute', escape_percentage, retval=True)
        event.listen(self.engine_daily_buy_list, 'before_execute', escape_percentage, retval=True)
        # 특정 데이터 베이스가 아닌, mysql 에 접속하는 객체
        self.db_conn = pymysql.connect(host=cf.db_ip, port=int(cf.db_port), user=cf.db_id, password=cf.db_passwd,
                                       charset='utf8')

    # 매수 함수
    def invest_send_order(self, date, code, code_name, price, yes_close, j):
        # print("invest_send_order!!!")
        # 시작가가 투자하려는 금액 보다 작아야 매수가 가능하기 때문에 아래 조건
        if price < self.invest_unit:
            print(code_name, " 매수!!!!!!!!!!!!!!!")

            # 매수를 하게 되면 all_item_db 테이블에 반영을 한다.
            self.db_to_all_item(date, self.df_realtime_daily_buy_list, j,
                                code,
                                code_name, price,
                                yes_close)

            # 매수를 성공적으로 했으면 realtime_daily_buy_list 테이블의 check_item 에 매수 시간을 설정
            self.update_realtime_daily_buy_list(code, date)

            # 일별, 분별 정산 함수
            self.check_balance()

    # code명으로 code_name을 가져오는 함수
    def get_name_by_code(self, code):

        sql = "select code_name from stock_item_all where code = '%s'"
        code_name = self.engine_daily_buy_list.execute(sql % (code)).fetchall()
        print(code_name)
        if code_name:
            return code_name[0][0]
        else:
            return False

    # 실제 매수하는 함수
    def auto_trade_stock_realtime(self, min_date, date_rows_today, date_rows_yesterday):
        print("auto_trade_stock_realtime 함수에 들어왔다!!")
        # self.df_realtime_daily_buy_list 에 있는 모든 종목들을 매수한다
        for j in range(self.len_df_realtime_daily_buy_list):
            if self.jango_check():

                # 종목 코드를 가져온다.
                code = str(self.df_realtime_daily_buy_list.loc[j, 'code']).rjust(6, "0")

                # 종목명을 가져온다.
                code_name = self.df_realtime_daily_buy_list.loc[j, 'code_name']

                # 매수 들어가기전에 db에 테이블이 존재하는지 확인
                # 분별 시뮬레이팅 인 경우
                if self.use_min:
                    # print("code_name!!", code_name)
                    # min_craw db에 종목이 없으면 매수 하지 않는다.
                    if not self.is_min_craw_table_exist(code_name):
                        continue
                # 일별 시뮬레이팅 인 경우
                else:
                    # daily_craw db에 종목이 없으면 매수 하지 않는다.
                    if not self.is_daily_craw_table_exist(code_name):
                        continue

                # open_price 를 가져오는 것을 분별/일별 시뮬레이션 구분하여 설정하였습니다.
                # 분별 시뮬레이션이 아닌 일별 시뮬레이션의 경우
                if not self.use_min:
                    # 매수 당일 시작가를 가져온다.
                    price = self.get_now_open_price_by_date(code, date_rows_today)
                # 분별 시뮬레이션의 경우
                else:
                    # 매수 시점의 가격을 가져온다.
                    price = self.get_now_close_price_by_min(code_name, min_date)

                # 어제 종가를 가져온다.
                yes_close = self.get_yes_close_price_by_date(code, date_rows_yesterday)

                # False는 데이터가 없는것
                if code_name == False or price == 0 or price == False:
                    continue

                # 아래 if 문 추가 (향후 실시간 조건 매수 시 사용) , if문에서 use_min, only_nine_buy,trade_check_num(1,2,3.../ 0만 False) 세가지 변수가 True이면 아래 함수 살행함
                if self.use_min and not self.only_nine_buy and self.trade_check_num:
                    # 시작가
                    open = self.get_now_open_price_by_date(code, date_rows_today)
                    # 당일 누적 거래량
                    sum_volume = self.get_now_volume_by_min(code_name, min_date)

                    # open, sum_volume 값이 존재 할 경우
                    if open and sum_volume:
                        # 매수 할 종목에 대한 dataframe row와, 시작가, 현재가, 분별 누적 거래량 정보를 전달
                        # loc[j] 에서 'j'는  for j in range(self.len_df_realtime_daily_buy_list): 에서 
                        # df_realtime_daily_buy_list 에 있는 모든 종목들을 하나 하나씩 가져오는 역할을 한다.
                        if not self.trade_check(self.df_realtime_daily_buy_list.loc[j], open, price, sum_volume):
                            # 실시간 매수 조건에 맞지 않는 경우 pass
                            # trade_check 함수에서 true 반환하면 if not 함수로 인해 false가 되므로 continue 되지 않고 
                            # 바로 invest_send_order 함수로 들어간다.
                            continue
                ################################################################

                # 매수 주문에 들어간다.
                self.invest_send_order(min_date, code, code_name, price, yes_close, j)
            else:
                break

    # 최근 daily_buy_list의 날짜 테이블에서 code에 해당 하는 row만 가져오는 함수
    def get_daily_buy_list_by_code(self, code, date):
        # print("get_daily_buy_list_by_code 함수에 들어왔습니다!")

        sql = "select * from `" + date + "` where code = '%s' group by code"

        daily_buy_list = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        df_daily_buy_list = DataFrame(daily_buy_list,
                                      columns=['index', 'index2', 'date', 'check_item',
                                               'code', 'code_name', 'd1_diff_rate', 'close', 'open',
                                               'high', 'low',
                                               'volume',
                                               'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
                                               'clo100', 'clo120',
                                               "clo5_diff_rate", "clo10_diff_rate", "clo20_diff_rate",
                                               "clo40_diff_rate", "clo60_diff_rate",
                                               "clo80_diff_rate", "clo100_diff_rate",
                                               "clo120_diff_rate",
                                               'yes_clo5', 'yes_clo10', 'yes_clo20', 'yes_clo40',
                                               'yes_clo60',
                                               'yes_clo80',
                                               'yes_clo100', 'yes_clo120',
                                               'vol5', 'vol10', 'vol20', 'vol40', 'vol60', 'vol80',
                                               'vol100', 'vol120'])
        return df_daily_buy_list

    # realtime_daily_buy_list 테이블의 매수 리스트를 가져오는 함수
    def get_realtime_daily_buy_list(self):
        print("get_realtime_daily_buy_list 함수에 들어왔습니다!")

        # 코드를 간소화 했습니다. 조건문 모두 없앴습니다.
        # check_item = 매수 했을 시 날짜가 찍혀 있다. 매수 하지 않았을 때는 0
        sql = "select * from realtime_daily_buy_list where check_item = '%s' group by code"

        realtime_daily_buy_list = self.engine_simulator.execute(sql % (0)).fetchall()

        self.df_realtime_daily_buy_list = DataFrame(realtime_daily_buy_list,
                                                    columns=['index', 'index2', 'index3', 'date', 'check_item',
                                                             'code', 'code_name', 'd1_diff_rate', 'close', 'open',
                                                             'high', 'low',
                                                             'volume',
                                                             'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
                                                             'clo100', 'clo120',
                                                             "clo5_diff_rate", "clo10_diff_rate", "clo20_diff_rate",
                                                             "clo40_diff_rate", "clo60_diff_rate",
                                                             "clo80_diff_rate", "clo100_diff_rate",
                                                             "clo120_diff_rate",
                                                             'yes_clo5', 'yes_clo10', 'yes_clo20', 'yes_clo40',
                                                             'yes_clo60',
                                                             'yes_clo80',
                                                             'yes_clo100', 'yes_clo120',
                                                             'vol5', 'vol10', 'vol20', 'vol40', 'vol60', 'vol80',
                                                             'vol100', 'vol120'])

        self.len_df_realtime_daily_buy_list = len(self.df_realtime_daily_buy_list)

    # 가장 최근의 daily_buy_list에 담겨 있는 날짜 테이블 이름을 가져오는 함수
    def get_recent_daily_buy_list_date(self):
        sql = "select TABLE_NAME from information_schema.tables where table_schema = 'daily_buy_list' and TABLE_NAME like '%s' order by table_name desc limit 1"
        row = self.engine_daily_buy_list.execute(sql % ("20%%")).fetchall()

        if len(row) == 0:
            return False
        return row[0][0]

    # 실시간 주가 분석 알고리즘 함수 
    def trade_check(self, df_row, open_price, current_price, current_sum_volume):
        '''
        :param df_row: 매수 종목 리스트(realtime_daily_buy_list)
        :param current_price: (현재가)
        :param current_sum_volume: (현재 누적 거래량)
        :return: True (매수), False(매수 X)
        '''
        code_name = df_row['code_name']
        yes_vol20 = df_row['vol20']
        yes_close = df_row['close']
        yes_high = df_row['high']
        yes_low = df_row['low']
        yes_volume = df_row['volume']

        # 실시간 거래 대금 체크 알고리즘
        if self.trade_check_num == 1:
            # 어제 거래 대금
            yes_total_tr_price = yes_close * yes_volume
            # 현재 거래 대금
            current_total_tr_price = current_price * current_sum_volume
            # 어제 종가 보다 현재가가 증가했고, 현재 거래 대금이 어제 거래대금에 비해서 x배 올라갔을 때 매수
            if current_price > yes_close and current_total_tr_price > yes_total_tr_price * self.volume_up:
                return True
            else:
                return False

        elif self.trade_check_num == 2:
            # 매수 가격 최저 범위
            min_buy_limit = int(yes_close) * self.invest_min_limit_rate
            # 매수 가격 최고 범위
            max_buy_limit = int(yes_close) * self.invest_limit_rate
            # 현재가가 매수 가격 최저 범위와 매수 가격 최고 범위 안에 들어와 있다면 매수 한다.
            if min_buy_limit < current_price < max_buy_limit:
                return True
            else:
                return False

        # 래리 윌리엄스 변동성 돌파 알고리즘(매수)
        elif self.trade_check_num == 3:
            # 변동폭(_range): 전일 고가(yes_high)에서 전일 저가(yes_low)를 뺀 가격/ Range = 전일 고가 - 전일 저가
            # 매수시점 : 현재가 > 시작가 + (변동폭 * k)  [k는 0~1 사이 수] / 진입 : 현재가 > 당일 시가 + Range*0.5 
            _range = yes_high - yes_low
            if open_price + _range * self.rarry_k < current_price:
                return True
            else:
                return False

        else:
            logger.debug("trade_check 함수에 self.trade_check_num = {} 에 맞는 알고리즘이 없습니다. ".format(self.trade_check_num))
            exit(1)


    

    # 여기서 sql문의 date는 반드시 어제 일자여야 한다. -> 어제 일자 기준 반영된 데이터로 종목을 선정해야함.
    ##!@####################################################################################################################################################################################
    # 매수 할 종목의 리스트를 선정 알고리즘
    def db_to_realtime_daily_buy_list(self, date_rows_today, date_rows_yesterday, i):
        # 5 / 20 골든크로스 buy
        if self.db_to_realtime_daily_buy_list_num == 1:
            # orderby는 거래량 많은 순서

            sql = "select a.* from `" + date_rows_yesterday + "` a where yes_clo20 > yes_clo5 and clo5 > clo20 " \
                                                            "and NOT exists (select null from stock_konex b where a.code=b.code) " \
                                                            "and close < '%s' group by code limit 1"
            realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.invest_unit)).fetchall()


        # 5 / 40 골든크로스 buy
        elif self.db_to_realtime_daily_buy_list_num == 2:
            # orderby는 거래량 많은 순서
            sql = "select a.* from `" + date_rows_yesterday + "` a where yes_clo40 > yes_clo5 and clo5 > clo40 " \
                                                            "and NOT exists (select null from stock_konex b where a.code=b.code) " \
                                                            "and close < '%s' group by code"
            realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.invest_unit)).fetchall()


        elif self.db_to_realtime_daily_buy_list_num == 3:
            sql = "select a.* from `" + date_rows_yesterday + "` a where d1_diff_rate > 1 " \
                                                            "and NOT exists (select null from stock_konex b where a.code=b.code) " \
                                                            "and close < '%s' group by code"
            # 아래 명령을 통해 테이블로 부터 데이터를 가져오면 리스트 형태로 realtime_daily_buy_list 에 담긴다.
            realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.invest_unit)).fetchall()

        
        # 5 / 60 골든크로스 buy
        elif self.db_to_realtime_daily_buy_list_num == 4:
            # orderby는 거래량 많은 순서  
            sql = "select a.* from `" + date_rows_yesterday + "` a where yes_clo60 > yes_clo5 and clo5 > clo60 " \
                                                            "and NOT exists (select null from stock_konex b where a.code=b.code) " \
                                                            "and close < '%s' group by code"
            realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.invest_unit)).fetchall()
        
        #관리, 불성실, 주의, 경고, 위험 제외 하고 buy
        elif self.db_to_realtime_daily_buy_list_num == 5:
            
            sql = "select a.* from `" + date_rows_yesterday + "` a " \
                    "where yes_clo20 > yes_clo5 and clo5 > clo20 " \
                    "and NOT exists (select null from stock_konex b where a.code=b.code)" \
                    "and NOT exists (select null from stock_managing c where a.code=c.code and c.code_name != '' group by c.code) " \
                    "and NOT exists (select null from stock_insincerity d where a.code=d.code and d.code_name !='' group by d.code) " \
                    "and NOT exists (select null from stock_invest_caution e where a.code=e.code and DATE_SUB('%s', INTERVAL '%s' MONTH ) < e.post_date and e.post_date < Date('%s') and e.type != '투자경고 지정해제' group by e.code)"\
                    "and NOT exists (select null from stock_invest_warning f where a.code=f.code and f.post_date <= DATE('%s') and (f.cleared_date > DATE('%s') or f.cleared_date is null) group by f.code)"\
                    "and NOT exists (select null from stock_invest_danger g where a.code=g.code and g.post_date <= DATE('%s') and (g.cleared_date > DATE('%s') or g.cleared_date is null) group by g.code)"\
                    "and a.close < '%s'"

            realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (date_rows_yesterday, self.interval_month, date_rows_yesterday,date_rows_yesterday ,date_rows_yesterday,date_rows_yesterday,date_rows_yesterday, self.invest_unit)).fetchall()

        # 5 / 20 골든크로스 적용되고
        # 관리, 불성실, 주의, 경고, 위험 제외 하고 buy
        # volume * close (총 거래대금 금액)이  self.total_transaction_price(변경가능) 원 보다 큰 주식을 buy 
        # 20일 평균거래량(거래량20이평선)대비 3배이상 거래량 터졌을때 강세( vol20 * '%s' < volume )이므로 buy
        # 전날보다  self.d1_diff 변수 값 (변경가능) 이상 올랐을 때 buy ( d1_diff_rate)
        ## 2022-10-01 Written by SEONGJAE-YOO   
        
        elif self.db_to_realtime_daily_buy_list_num == 6:
            
            sql = "select a.* from `" + date_rows_yesterday + "` a " \
                    "where yes_clo20 > yes_clo5 and clo5 > clo20 " \
                    "and volume * close > '%s' " \
                    "and vol20 * '%s' < volume " \
                    "and d1_diff_rate > '%s' " \
                    "and NOT exists (select null from stock_konex b where a.code=b.code) " \
                    "and NOT exists (select null from stock_managing c where a.code=c.code and c.code_name != '' group by c.code) " \
                    "and NOT exists (select null from stock_insincerity d where a.code=d.code and d.code_name !='' group by d.code) " \
                    "and NOT exists (select null from stock_invest_caution e where a.code=e.code and DATE_SUB('%s', INTERVAL '%s' MONTH ) < e.post_date and e.post_date < Date('%s') and e.type != '투자경고 지정해제' group by e.code) " \
                    "and NOT exists (select null from stock_invest_warning f where a.code=f.code and f.post_date <= DATE('%s') and (f.cleared_date > DATE('%s') or f.cleared_date is null) group by f.code) " \
                    "and NOT exists (select null from stock_invest_danger g where a.code=g.code and g.post_date <= DATE('%s') and (g.cleared_date > DATE('%s') or g.cleared_date is null) group by g.code) " \
                    "and a.close < '%s' " \
                    "order by volume * close desc "

            realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.total_transaction_price,self.vol_mul, self.d1_diff, date_rows_yesterday, self.interval_month, date_rows_yesterday,date_rows_yesterday ,date_rows_yesterday,date_rows_yesterday,date_rows_yesterday, self.invest_unit)).fetchall()

    # Absolute Momentum 전략 : 특정일 전의 종가 보다 n% 이상 상승한 종목 매수 (code version)
    # 2022-10-04 Written by SEONGJAE-YOO  (Commits on october 4, 2022)
        elif self.db_to_realtime_daily_buy_list_num == 7:
            # 아래에서 필터링 된 매수종목을 append 해주기 위해 비어있는 리스트를 만들어준다.
            realtime_daily_buy_list = []
            if i < self.day_before + 1:
                pass
            else:
                sql = "SELECT YES_DAY.* FROM `" + date_rows_yesterday +"` YES_DAY " \
                       "WHERE NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code) " \
                       "AND close < '%s' "
                # realtime_daily_buy_list_temp 로 일단 위 조건의 종목을을받는다.
                realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql % (self.invest_unit)).fetchall()
                for row in realtime_daily_buy_list_temp:
                    # 종목코드
                    code = row[4]
                    # 어제 종가
                    yes_close = row[7]
                    # date_rows_yesterday 가 self.date_rows[i-1] 값이다.
                    # 어제 일자 기준 n 일전 날짜
                    date_before = self.date_rows[i-1-self.day_before][0]
                    # 어제 일자 기준 n 일전 종가
                    date_before_close = self.get_now_close_price_by_date(code, date_before)
                    if date_before_close != 0 and date_before_close != False :
                        # 모멘텀 계산 : n일전 종가 대비 수익률
                        diff_point_calc = (yes_close - date_before_close) / date_before_close * 100
                        # 모멘텀(수익률)이 self.diff_point 보다 높을 경우 realtime_daily_buy_list에 append
                        if diff_point_calc > self.diff_point:
                            realtime_daily_buy_list.append(row)
        
        
        # Absolute Momentum 전략 : 특정일 전의 종가 보다 n% 이상 상승한 종목 매수 (query vesrion)
        # date_before 테이블은 BEFORE_DAY 와 같다 , date_rows_yesterday 테이블은 YES_DAY 와 같다 
        # BEFORE_DAY.code = YES_DAY.code 은 조인 이다.(즉, 두개 이상의 테이블에 대해서 결합하여 나타낼 때 )
        elif self.db_to_realtime_daily_buy_list_num == 8:
            if i < self.day_before + 1:
                realtime_daily_buy_list = []
                pass
            else:
                date_before = self.date_rows[i - 1 - self.day_before][0]
                sql = "SELECT YES_DAY.* " \
                      "FROM `"+date_before+"` BEFORE_DAY, `" + date_rows_yesterday +"` YES_DAY " \
                        "WHERE BEFORE_DAY.code = YES_DAY.code " \
                        "AND (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 > '%s' " \
                        "AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code)" \
                        "AND YES_DAY.close < '%s'"
        
                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.diff_point, self.invest_unit)).fetchall()
        
        # 상대 모멘텀 전략(Relative Strength Momentum) : 특정일 전의 종가 보다 n% 이상 상승한 종목 중 가장 많이 상승한 종목 순으로 매수 (내림차순) (query version)
        elif self.db_to_realtime_daily_buy_list_num == 9:
            if i < self.day_before + 1:
                realtime_daily_buy_list = []
                pass
            else:
                date_before = self.date_rows[i - 1 - self.day_before][0]
                sql = "SELECT YES_DAY.* " \
                      "FROM `" + date_before + "` BEFORE_DAY, `" + date_rows_yesterday + "` YES_DAY " \
                     "WHERE BEFORE_DAY.code = YES_DAY.code " \
                     "AND (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 > '%s' " \
                     "AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code)" \
                     "AND YES_DAY.close < '%s'" \
                     "ORDER BY (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 DESC"
        
                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.diff_point, self.invest_unit)).fetchall()

        # 2022-10-08 Written by SEONGJAE-YOO (Commits on Oct 8, 2022)
        elif self.db_to_realtime_daily_buy_list_num == 10:
            if i < self.day_before + 1:
                realtime_daily_buy_list = []
                pass
            else:
                date_before = self.date_rows[i - 1 - self.day_before][0]
                sql = "SELECT YES_DAY.* " \
                    "FROM `" + date_before + "` BEFORE_DAY, `" + date_rows_yesterday + "` YES_DAY " \
                    "WHERE BEFORE_DAY.code = YES_DAY.code " \
                    "and YES_DAY.yes_clo20 > YES_DAY.yes_clo5 and YES_DAY.clo5 > YES_DAY.clo20 " \
                    "and YES_DAY.volume * YES_DAY.close > '%s' " \
                    "and YES_DAY.vol20 * '%s' < YES_DAY.volume " \
                    "and YES_DAY.d1_diff_rate > '%s' " \
                    "and NOT exists (select null from stock_konex b where YES_DAY.code=b.code) " \
                    "and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) " \
                    "and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) " \
                    "and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB('%s', INTERVAL '%s' MONTH ) < e.post_date and e.post_date < Date('%s') and e.type != '투자경고 지정해제' group by e.code) " \
                    "and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE('%s') and (f.cleared_date > DATE('%s') or f.cleared_date is null) group by f.code) " \
                    "and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE('%s') and (g.cleared_date > DATE('%s') or g.cleared_date is null) group by g.code) " \
                    "AND (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 > '%s' " \
                    "AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code)" \
                    "AND YES_DAY.close < '%s'" \
                    "ORDER BY (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 DESC"

                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql % (self.total_transaction_price,self.vol_mul, self.d1_diff, date_rows_yesterday, self.interval_month, date_rows_yesterday,date_rows_yesterday ,date_rows_yesterday,date_rows_yesterday,date_rows_yesterday,self.diff_point, self.invest_unit)).fetchall()

        elif self.db_to_realtime_daily_buy_list_num == 11:

                sql = f'''
                    SELECT day.* FROM `{date_rows_yesterday}` day, stock_info info
                    WHERE day.code = info.code
                    AND info.stock_market IN ("거래소", "코스닥")
                    AND info.category0 IN ("우량기업", "신성장기업")
                    AND info.audit = '정상'
                    AND info.margin <= 40
                    AND info.remarks NOT LIKE "%관리종목%"
                    AND info.remarks NOT LIKE "%거래정지%"
                '''
                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()

        # 2022-10-14 Written by SEONGJAE-YOO (Commits on Oct 14, 2022)
        elif self.db_to_realtime_daily_buy_list_num == 12:
            if i < self.day_before + 1:
                realtime_daily_buy_list = []
                pass
            else:
                date_before = self.date_rows[i - 1 - self.day_before][0]
                sql = f'''
                    SELECT YES_DAY.* FROM `{date_before}` BEFORE_DAY, `{date_rows_yesterday}` YES_DAY, stock_info info 
                    WHERE BEFORE_DAY.code = YES_DAY.code
                    AND YES_DAY.code = info.code  
                    and YES_DAY.yes_clo20 > YES_DAY.yes_clo5 and YES_DAY.clo5 > YES_DAY.clo20  
                    and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                    and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                    and YES_DAY.d1_diff_rate > {self.d1_diff}  
                    and info.audit = '{self.audit}'  
                    and info.margin <= {self.margin} 
                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                    and info.remarks NOT LIKE '{self.remarks_stop}'  
                    and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                    and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                    and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                    and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                    and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                    AND (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 > {self.diff_point}
                    AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code) 
                    AND YES_DAY.close < {self.invest_unit}
                    ORDER BY (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 DESC
                '''
                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()


        # 2022-10-18 Written by SEONGJAE-YOO (Commits on Oct 18, 2022)
        elif self.db_to_realtime_daily_buy_list_num == 13:
            if i < self.day_before + 1:
                realtime_daily_buy_list = []
                pass
            else:
                date_before = self.date_rows[i - 1 - self.day_before][0]
                sql = f'''
                    SELECT YES_DAY.* FROM `{date_before}` BEFORE_DAY, `{date_rows_yesterday}` YES_DAY, stock_info info 
                    WHERE BEFORE_DAY.code = YES_DAY.code
                    AND YES_DAY.code = info.code  
                    and YES_DAY.yes_clo20 > YES_DAY.yes_clo5 and YES_DAY.clo5 > YES_DAY.clo20  
                    and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                    and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                    and YES_DAY.d1_diff_rate > {self.d1_diff}
                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_b}')
                    and info.category0 IN ('{self.category0_a}', '{self.category0_b}','{self.category0_c}')  
                    and info.audit = '{self.audit}'  
                    and info.margin <= {self.margin} 
                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                    and info.remarks NOT LIKE '{self.remarks_stop}'  
                    and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                    and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                    and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                    and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                    and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                    AND (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 > {self.diff_point}
                    AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code) 
                    ORDER BY (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 DESC
                '''
                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()

                # AND YES_DAY.close < {self.invest_unit}

        # 볼린저밴드 알고리즘을 위해 추가된 부분 (매수 리스트 알고리즘)
       
        elif self.db_to_realtime_daily_buy_list_num == 14:
            # 20일 이동평균선과 승수 2 사용
            realtime_daily_buy_list = []
            # 사용하는 이동평균선 기간
            ma_period = 20

            if i > ma_period:
                '''
                아래 쿼리에 'ORDER BY volume * close DESC limit 100' 조건은 전종목 (2000여개)을 검색할 경우 시뮬레이션 하루당 80여초가량 소모되어  
                거래대금으로 나열한 뒤 100개의 종목들만 가져오도록 제한하였습니다.
                차후 실제로 알고리즘을 구현하실 때에는 보조지표 중의 하나인 볼린저밴드를 이용하시기 전에 충분한 조건들을 추가하여 종목 수를 제한하는 것을 추천드립니다. 
                (예시. 이평선, 거래량 사용)
                '''
                sql = f"""
                        SELECT YES_DAY.*
                        FROM `{date_rows_yesterday}` YES_DAY
                        WHERE NOT exists(SELECT null FROM stock_konex b WHERE YES_DAY.code = b.code)
                            AND  volume != 0 
                            AND close < '{self.invest_unit}'
                            ORDER BY volume * close DESC limit 100
                    """
                realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql).fetchall()

                # 과매도 포지션 포착
                for item in realtime_daily_buy_list_temp:
                    code_name = item.code_name
                    # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                    bb_sql = f"""
                            SELECT close
                            FROM `{code_name}`
                            WHERE date <= '{date_rows_yesterday}'
                            ORDER BY date DESC limit {ma_period}
                        """
                    df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                    if len(df_close) >= ma_period:
                        # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                        result = BBands(pd.DataFrame(df_close), w=ma_period)
                        # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                        if result:
                            mbb, ubb, lbb, perb, bw = result
                            # perb가 음수인 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                            if perb < 0:
                                realtime_daily_buy_list.append(item)



        elif self.db_to_realtime_daily_buy_list_num == 15:
                    
                    ma_period = 20
                    if i < self.day_before + 1:
                        realtime_daily_buy_list = []
                        pass
                    else:
                        date_before = self.date_rows[i - 1 - self.day_before][0]
                        if i > ma_period:
                                sql = f'''
                                    SELECT YES_DAY.* FROM `{date_before}` BEFORE_DAY, `{date_rows_yesterday}` YES_DAY, stock_info info 
                                    WHERE BEFORE_DAY.code = YES_DAY.code
                                    AND YES_DAY.code = info.code  
                                    and YES_DAY.yes_clo20 > YES_DAY.yes_clo5 and YES_DAY.clo5 > YES_DAY.clo20  
                                    and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                                    and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                                    and YES_DAY.d1_diff_rate > {self.d1_diff}  
                                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_b}', '{self.stock_market_c}') 
                                    and info.category0 IN ('{self.category0_a}', '{self.category0_b}','{self.category0_c}')  
                                    and info.audit = '{self.audit}'  
                                    and info.margin <= {self.margin} 
                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                    and info.remarks NOT LIKE '{self.remarks_stop}'  
                                    and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                    and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                    and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                    and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                    and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                                    AND (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 > {self.diff_point}
                                    AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code) 
                                    AND YES_DAY.close < {self.invest_unit}
                                    ORDER BY (YES_DAY.close - BEFORE_DAY.close) / BEFORE_DAY.close * 100 DESC
                                '''
                                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()

                                # 과매도 포지션 포착
                                for item in realtime_daily_buy_list:
                                    code_name = item.code_name
                                    # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                    bb_sql = f"""
                                            SELECT close
                                            FROM `{code_name}`
                                            WHERE date <= '{date_rows_yesterday}'
                                            ORDER BY date DESC limit {ma_period}
                                        """
                                    df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                    if len(df_close) >= ma_period:
                                        # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                        result = BBands(pd.DataFrame(df_close), w=ma_period)
                                        # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                        if result:
                                            mbb, ubb, lbb, perb, bw = result
                                            # perb가 음수인 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                            if perb < 0:
                                                realtime_daily_buy_list.append(item)


        elif self.db_to_realtime_daily_buy_list_num == 16:
                            
                    ma_period = 20
                    if i < self.date_before_b + 1:
                        realtime_daily_buy_list = []
                        pass
                    else:
                        date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                        date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                        
                        if i > ma_period:
                                sql = f'''
                                    SELECT YES_DAY.* FROM daily_buy_list.`{date_before_a}` BEFORE_DAY_A, daily_buy_list.`{date_before_b}` BEFORE_DAY_B, `{date_rows_yesterday}` YES_DAY, stock_info info 
                                    WHERE BEFORE_DAY_A.code = BEFORE_DAY_B.code
                                    AND BEFORE_DAY_B.code = YES_DAY.code
                                    AND YES_DAY.code = info.code  
                                    and YES_DAY.yes_clo20 > YES_DAY.yes_clo5 and YES_DAY.clo5 > YES_DAY.clo20  
                                    and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                                    and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_b}', '{self.stock_market_c}') 
                                    and info.category0 IN ('{self.category0_a}', '{self.category0_b}','{self.category0_c}')  
                                    and info.audit = '{self.audit}'  
                                    and info.margin <= {self.margin} 
                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                    and info.remarks NOT LIKE '{self.remarks_stop}'  
                                    and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                    and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                    and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                    and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                    and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                                    AND (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100)) / 2 > {self.diff_point}
                                    AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code) 
                                    AND YES_DAY.close < {self.invest_unit}
                                    ORDER BY (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100)) / 2 DESC
                                '''
                                realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()

                                # 과매도 포지션 포착
                                for item in realtime_daily_buy_list:
                                    code_name = item.code_name
                                    # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                    bb_sql = f"""
                                            SELECT close
                                            FROM `{code_name}`
                                            WHERE date <= '{date_rows_yesterday}'
                                            ORDER BY date DESC limit {ma_period}
                                        """
                                    df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                    if len(df_close) >= ma_period:
                                        # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                        result = BBands(pd.DataFrame(df_close), w=ma_period)
                                        # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                        if result:
                                            mbb, ubb, lbb, perb, bw = result
                                            # perb가 음수인 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                            if perb < 0:
                                                realtime_daily_buy_list.append(item)


       # 상대 모멘텀 전략(Relative Strength Momentum) : 특정일 전의 종가 보다 n% 이상 상승한 종목 중 가장 많이 상승한 종목 순으로 매수 (내림차순)
        elif self.db_to_realtime_daily_buy_list_num == 17:
                                
                        ma_period = 20
                        if i < self.date_before_d + 1:
                            realtime_daily_buy_list = []
                            pass
                        else:
                            date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            date_before_d =self.date_rows[i - 1 - self.date_before_d][0]

                                

                            if i > ma_period:
                                    sql = f'''
                                        SELECT YES_DAY.* FROM `{date_before_a}` BEFORE_DAY_A, `{date_before_b}` BEFORE_DAY_B, `{date_before_c}` BEFORE_DAY_C, `{date_before_d}` BEFORE_DAY_D, `{date_rows_yesterday}` YES_DAY, stock_info info 
                                        WHERE BEFORE_DAY_A.code = BEFORE_DAY_B.code
                                        AND BEFORE_DAY_B.code = BEFORE_DAY_C.code
                                        AND BEFORE_DAY_C.code = BEFORE_DAY_D.code 
                                        AND BEFORE_DAY_D.code = YES_DAY.code
                                        AND YES_DAY.code = info.code  
                                        and YES_DAY.clo5 > YES_DAY.clo20  
                                        and info.stock_market IN ('{self.stock_market_a}') 
                                        and info.audit = '{self.audit}'  
                                        and info.remarks NOT LIKE '{self.remarks_manage}'  
                                        and info.remarks NOT LIKE '{self.remarks_stop}'  
                                        and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                        and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                        and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                        and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                        and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                                        AND (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 > {self.diff_point}
                                        AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code) 
                                        AND YES_DAY.close < {self.invest_unit}
                                        ORDER BY (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 DESC
                                    '''
                                    realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()
                                    #and info.category0 IN ('{self.category0_a}', '{self.category0_b}','{self.category0_c}')  
                                    # and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                                    # and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                                    #and info.margin <= {self.margin} 
                                    # 과매도 포지션 포착
                                    for item in realtime_daily_buy_list:
                                        code_name = item.code_name
                                        # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                        bb_sql = f"""
                                                SELECT close
                                                FROM `{code_name}`
                                                WHERE date <= '{date_rows_yesterday}'
                                                ORDER BY date DESC limit {ma_period}
                                            """
                                        df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                        if len(df_close) >= ma_period:
                                            # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                            result = BBands(pd.DataFrame(df_close), w=ma_period)
                                            # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                            if result:
                                                mbb, ubb, lbb, perb, bw = result
                                                # # perb가 음수인 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                                # if perb < 0:
                                                #     realtime_daily_buy_list.append(item)
                                                #perb가 양수또는 0인 경우: 종가가 볼린저밴드 하한선보다 위에 위치 할 경우 매수리스트에서 제외한다
                                                if perb >= 0:
                                                    realtime_daily_buy_list.remove(item)
                                                    
        # 상대 모멘텀 전략(Relative Strength Momentum) : 특정일 전의 종가 보다 n% 이상 상승한 종목 중 가장 많이 상승한 종목 순으로 매수 (내림차순)
        #best
        elif self.db_to_realtime_daily_buy_list_num == 18:
                                
                        ma_period = 70 
                        # 3개월 기준
                        # (60,2.2) -> , 1970127 //  (60,2.3) -> 1970127, (60,1.8)- >1970127,  (70,3.0) 2201207, 
                        
                        realtime_daily_buy_list = []
                        if i < self.date_before_d + 1:
                            
                            pass
                        else:
                            date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            date_before_d =self.date_rows[i - 1 - self.date_before_d][0]

                            date_before_e = self.date_rows[i -1 -1][0]      

                            if i > ma_period:
                                    sql = f'''
                                        SELECT YES_DAY.* FROM `{date_before_a}` BEFORE_DAY_A, `{date_before_b}` BEFORE_DAY_B, `{date_before_c}` BEFORE_DAY_C, `{date_before_d}` BEFORE_DAY_D, `{date_before_e}` BEFORE_DAY_E,`{date_rows_yesterday}` YES_DAY, stock_info info 
                                        WHERE BEFORE_DAY_A.code = BEFORE_DAY_B.code
                                        AND BEFORE_DAY_B.code = BEFORE_DAY_C.code
                                        AND BEFORE_DAY_C.code = BEFORE_DAY_D.code 
                                        AND BEFORE_DAY_D.code = BEFORE_DAY_E.code
                                        AND BEFORE_DAY_E.code = YES_DAY.code
                                        AND YES_DAY.code = info.code  
                                        and YES_DAY.clo5 > YES_DAY.clo20     
                                        and info.audit = '{self.audit}'  
                                        and info.remarks NOT LIKE '{self.remarks_manage}'  
                                        and info.remarks NOT LIKE '{self.remarks_stop}'  
                                        and NOT exists (select * from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                        and NOT exists (select * from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                        and NOT exists (select * from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                        and NOT exists (select * from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                        and NOT exists (select * from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                                        AND YES_DAY.close < {self.invest_unit}
                                        AND (YES_DAY.close + ((BEFORE_DAY_E.high - BEFORE_DAY_E.low) * 0.5)) < BEFORE_DAY_E.open
                                        AND (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 > {self.diff_point}
                                        ORDER BY (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 DESC
                                    '''
                                    realtime_daily_buy_list_1 = self.engine_daily_buy_list.execute(sql).fetchall()
                                    # AND NOT exists (SELECT * FROM stock_konex b WHERE YES_DAY.code=b.code)
                                    # and info.stock_market IN ('{self.stock_market_a}','{self.stock_market_c}')
                                    #and info.category0 IN ('{self.category0_a}', '{self.category0_b}','{self.category0_c}')  
                                    # and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                                    # and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                                    #and info.margin <= {self.margin} 
                                    #AND exists (SELECT * FROM stock_etf ETF WHERE YES_DAY.code=ETF.code)
                                    #AND exists (SELECT * FROM stock_kospi d WHERE YES_DAY.code=d.code)
                                    # 과매도 포지션 포착
                                    for item in realtime_daily_buy_list_1:
                                        code_name = item.code_name
                                        # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                        bb_sql = f"""
                                                SELECT close
                                                FROM `{code_name}`
                                                WHERE date <= '{date_rows_yesterday}'
                                                ORDER BY date DESC limit {ma_period}
                                            """
                                        df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                        if len(df_close) >= ma_period:
                                            # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                            result = BBands(pd.DataFrame(df_close), w=ma_period)
                                            # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                            if result:
                                                mbb, ubb, lbb, perb, bw = result
                                                # # perb가 0.5 보다 작을 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                                 
                                               
                                                if perb <= 0.5:
                                                    
                                                    realtime_daily_buy_list.append(item)
                                                #perb가 양수또는 0인 경우: 종가가 볼린저밴드 하한선보다 위에 위치 할 경우 매수리스트에서 제외한다
                                                # if perb >= 0:
                                                #     realtime_daily_buy_list.remove(item)   
                                    #realtime_daily_buy_list.append(realtime_daily_buy_list_DB)    


        elif self.db_to_realtime_daily_buy_list_num == 19:
                                
                        ma_period = 20
                        if i < self.date_before_d + 1:
                            realtime_daily_buy_list = []
                            pass
                        else:
                            date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            date_before_d =self.date_rows[i - 1 - self.date_before_d][0]

                            date_before_e = self.date_rows[i -1 -1][0]      

                            if i > ma_period:
                                    sql = f'''
                                        SELECT YES_DAY.* FROM `{date_before_a}` BEFORE_DAY_A, `{date_before_b}` BEFORE_DAY_B, `{date_before_c}` BEFORE_DAY_C, `{date_before_d}` BEFORE_DAY_D, `{date_before_e}` BEFORE_DAY_E,`{date_rows_yesterday}` YES_DAY, stock_info info 
                                        WHERE BEFORE_DAY_A.code = BEFORE_DAY_B.code
                                        AND BEFORE_DAY_B.code = BEFORE_DAY_C.code
                                        AND BEFORE_DAY_C.code = BEFORE_DAY_D.code 
                                        AND BEFORE_DAY_D.code = BEFORE_DAY_E.code
                                        AND BEFORE_DAY_E.code = YES_DAY.code
                                        AND YES_DAY.code = info.code  
                                        and YES_DAY.clo5 > YES_DAY.clo20    
                                        and info.stock_market IN ('{self.stock_market_a}') 
                                        and info.audit = '{self.audit}'  
                                        and info.remarks NOT LIKE '{self.remarks_manage}'  
                                        and info.remarks NOT LIKE '{self.remarks_stop}'  
                                        and NOT exists (select null from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                        and NOT exists (select null from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                        and NOT exists (select null from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                        and NOT exists (select null from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                        and NOT exists (select null from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code) 
                                        AND NOT exists (SELECT null FROM stock_konex b WHERE YES_DAY.code=b.code)
                                        AND (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 > {self.diff_point}
                                        AND (YES_DAY.close + ((BEFORE_DAY_E.high - BEFORE_DAY_E.low) * 0.5)) < BEFORE_DAY_E.open
                                        AND YES_DAY.close < {self.invest_unit}
                                        ORDER BY (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 DESC
                                    '''
                                    realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()

        elif self.db_to_realtime_daily_buy_list_num == 20:
                                
                        ma_period = 70 
                        realtime_daily_buy_list = []
                        realtime_daily_buy_list_1 = []
                        realtime_daily_buy_list_2 = []
                       # realtime_daily_buy_list_3 = []
                        if i < self.date_before_d + 1:
                            
                            pass
                        
                        else:        
                            # date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            # date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            # date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            # date_before_d =self.date_rows[i - 1 - self.date_before_d][0]
    
                            # date_before_e = self.date_rows[i -1 -1][0]      
                            
                            #if i > ma_period:
                                            

                                            sql = f"""
                                                    SELECT YES_DAY.*
                                                    FROM `{date_rows_yesterday}` YES_DAY ,stock_info info 
                                                    WHERE YES_DAY.code = info.code 
                                                    and info.audit = '{self.audit}'  
                                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                                    and info.remarks NOT LIKE '{self.remarks_stop}'  
                                                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_c}')   
                                                    and NOT exists (select * from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                                    and NOT exists (select * from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                                    and NOT exists (select * from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                                    and NOT exists (select * from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                                    and NOT exists (select * from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code)  
                                                    AND YES_DAY.volume != 0 
                                                    AND YES_DAY.close < {self.invest_unit}
                                                    ORDER BY YES_DAY.volume * YES_DAY.close DESC 
                                                """
                                            realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql).fetchall()

                                            # 과매도 포지션 포착
                                            for item in realtime_daily_buy_list_temp:
                                                code_name = item.code_name
                                                # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                                bb_sql = f"""
                                                        SELECT close
                                                        FROM `{code_name}`
                                                        WHERE date <= '{date_rows_yesterday}'
                                                        ORDER BY date DESC limit {ma_period}
                                                    """
                                                df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                                if len(df_close) >= ma_period:
                                                    # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                                    result = BBands(pd.DataFrame(df_close), w=ma_period)
                                                    # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                                    if result:
                                                        mbb, ubb, lbb, perb, bw = result
                                                        # perb가 음수인 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                                        if perb < 0.5:
                                                            items_code_name = item.code_name
                                                            items = f"""
                                                                    SELECT *
                                                                    FROM `{date_rows_yesterday}` 
                                                                    WHERE  code_name = '{items_code_name}'
                                                                    """
                                                            items = self.engine_daily_buy_list.execute(items).fetchall()
                                            
                                        

                                                            # 종목코드
                                                            code = items[0][4]
                                                            #code_name = row[5]
                                                            # 어제 종가
                                                            yes_close = items[0][7]

                                                            yes_open = items[0][8]
                                                            # date_rows_yesterday 가 self.date_rows[i-1] 값이다.
                                                            # 어제 일자 기준 n 일전 날짜
                                                            #date_before = self.date_rows[i-1-self.day_before][0]
                                                            
                                                            date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                                                            date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                                                            date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                                                            date_before_d =self.date_rows[i - 1 - self.date_before_d][0]
                                                            date_before_e =self.date_rows[i - 1 - 5][0]
                                                            
                                                            
                                                            # 어제 일자 기준 n 일전 종가
                                                            #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                                                            date_before_close_a = self.get_now_close_price_by_date(code, date_before_a)
                                                            date_before_close_b = self.get_now_close_price_by_date(code, date_before_b)
                                                            date_before_close_c = self.get_now_close_price_by_date(code, date_before_c)
                                                            date_before_close_d = self.get_now_close_price_by_date(code, date_before_d)
                                                            date_before_close_e = self.get_now_close_price_by_date(code, date_before_e)
                                                            
                                                            # self.date_before_a = 20 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                                                            # # n일 전 종가 대비 현재 종가(현재가)가 몇 프로 증가 했을 때 매수, 몇 프로 떨어졌을 때 매도 할 지 설정(0으로 설정 시 단순히 증가 했을 때 매수, 감소 했을 때 매도)
                                                            # self.date_before_b = 59 # 단위 일 (모멘텀에서 현재가랑 몇 일전의 종가와 비교할지)
                                                            # self.date_before_c = 123
                                                            # self.date_before_d = 245
                                                            # self.day_before = 20        
                                                            

                                                            # date_before_high_e = self.get_high_price_by_date(code, date_before_e)
                                                            # date_before_low_e = self.get_low_price_by_date(code, date_before_e)
                                                            # date_before_open_e = self.get_open_price_by_date(code, date_before_e)

                                                            if date_before_close_a != 0 and date_before_close_a != False and date_before_close_b != 0 and date_before_close_b != False and date_before_close_c != 0 and date_before_close_c != False and date_before_close_d != 0 and date_before_close_d != False: 
                                                                # 모멘텀 계산 : n일전 종가 대비 수익률
                                                                diff_point_calc = (((yes_close - date_before_close_a) / date_before_close_a * 100) +
                                                                                    ((yes_close - date_before_close_b) / date_before_close_b * 100) +
                                                                                    ((yes_close - date_before_close_c) / date_before_close_c * 100) + 
                                                                                    ((yes_close - date_before_close_d) / date_before_close_d * 100)) / 4 

                                                                diff_point_calc_2 = (((yes_close - date_before_close_c) / date_before_close_c * 100) +
                                                                                    ((yes_close - date_before_close_d) / date_before_close_d * 100)) / 2 

                                                                diff_point_calc_3 = (((yes_close - date_before_close_a) / date_before_close_a * 100) + ((yes_close - date_before_close_b) / date_before_close_b * 100) + ((yes_close - date_before_close_c) / date_before_close_c * 100)) / 3                                                                                                  

                                                                
                                                                diff_point_calc_4 = (((yes_close - date_before_close_a) / date_before_close_a * 100) +
                                                                                    ((yes_close - date_before_close_b) / date_before_close_b * 100) +
                                                                                    ((yes_close - date_before_close_c) / date_before_close_c * 100) + 
                                                                                    ((yes_close - date_before_close_d) / date_before_close_d * 100) +
                                                                                    ((yes_close - date_before_close_e) / date_before_close_e * 100)) / 5 

                                                                                                  

                                                                # 모멘텀(수익률)이 self.diff_point 보다 높을 경우 realtime_daily_buy_list에 append
                                                                #if ((diff_point_calc > self.diff_point) and (diff_point_calc_2 > self.diff_point)) or ((diff_point_calc_3 > self.diff_point) and (diff_point_calc_4 >self.diff_point)):                                                           
                                                                   # realtime_daily_buy_list_2.append(items)

                                                                if   ((diff_point_calc + diff_point_calc_2 + diff_point_calc_3 + diff_point_calc_4) / 4) >  self.diff_point: 
                                                                    #code_name = str(row.code_name)
                                                                    #code  =  int(row.code)                                  
                                                                    sql = f"""
                                                                        SELECT *
                                                                        FROM `{date_rows_yesterday}`                                                                    
                                                                        where code = {code}
                                                                        """
                                                                    realtime_daily_buy_list += self.engine_daily_buy_list.execute(sql).fetchall()
                                                                    # realtime_daily_buy_list.append(realtime_daily_buy_list_2)    
                                                        # code_name = '{code_name}'
                                                                
                                                        
                                                        
                                                        # realtime_daily_buy_list_1.append(item)
                                                        
                                
                                                        # code = item.code
                                                        # sql = f'''
                                                        #     SELECT YES_DAY.* FROM `{date_before_a}` BEFORE_DAY_A, `{date_before_b}` BEFORE_DAY_B, `{date_before_c}` BEFORE_DAY_C, `{date_before_d}` BEFORE_DAY_D, `{date_before_e}` BEFORE_DAY_E,`{date_rows_yesterday}` YES_DAY
                                                        #     WHERE BEFORE_DAY_A.code = BEFORE_DAY_B.code
                                                        #     AND BEFORE_DAY_B.code = BEFORE_DAY_C.code
                                                        #     AND BEFORE_DAY_C.code = BEFORE_DAY_D.code 
                                                        #     AND BEFORE_DAY_D.code = BEFORE_DAY_E.code
                                                        #     AND BEFORE_DAY_E.code = YES_DAY.code
                                                        #     AND YES_DAY.code = '{code}'
                                                            #     AND (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 > {self.diff_point}
                                                            #     ORDER BY (((YES_DAY.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((YES_DAY.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((YES_DAY.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((YES_DAY.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 DESC

                                                            # '''                                                                   
                                                            # realtime_daily_buy_list = self.engine_daily_buy_list.execute(sql).fetchall()

                                                            #realtime_daily_buy_list.append(realtime_daily_buy_list_1)                        
                                                            #     sql = f'''
                                                            #         SELECT YES_DAY.* 
                                                            #         FROM `{code}` YES_DAY
                                                            #         WHERE YES_DAY.date <= '{date_rows_yesterday}'

                                                            #         AND YES_DAY.close + (DATE_SUB({date_rows_yesterday}, INTERVAL 1 DAY)).high - (DATE_SUB({date_rows_yesterday}, INTERVAL 1 DAY).low) * 0.5) < (DATE_SUB({date_rows_yesterday}, INTERVAL 1 DAY).open
                                                                    
                                                                    
                                                            
                                                            #     '''
                                                            #     realtime_daily_buy_list = self.engine_daily_craw.execute(sql).fetchall()
                                                            # # AND (((YES_DAY.close - {date_rows_yesterday-20}.close) / {date_rows_yesterday-20}.close * 100) + ((YES_DAY.close - {date_rows_yesterday-60}.close) / {date_rows_yesterday-60}.close * 100) + ((YES_DAY.close - {date_rows_yesterday-120}.close) / {date_rows_yesterday-120}.close * 100) + ((YES_DAY.close - {date_rows_yesterday-240}.close) / {date_rows_yesterday-240}.close * 100)) / 4 > {self.diff_point}
                                                            #     ORDER BY (((YES_DAY.close - {date_rows_yesterday-20}.close) / {date_rows_yesterday-20}.close * 100) + ((YES_DAY.close - {date_rows_yesterday-60}.close) / {date_rows_yesterday-60}.close * 100) + ((YES_DAY.close - {date_rows_yesterday-120}.close) / {date_rows_yesterday-120}.close * 100) + ((YES_DAY.close - {date_rows_yesterday-240}.close) / {date_rows_yesterday-240}.close * 100)) / 4 DESC
                                                            
                                                            # AND NOT exists (SELECT * FROM stock_konex b WHERE YES_DAY.code=b.code)
                                                            # and info.stock_market IN ('{self.stock_market_a}','{self.stock_market_c}')
                                                            #and info.category0 IN ('{self.category0_a}', '{self.category0_b}','{self.category0_c}')  
                                                            # and YES_DAY.volume * YES_DAY.close > {self.total_transaction_price}  
                                                            # and YES_DAY.vol20 * {self.vol_mul} < YES_DAY.volume  
                                                            #and info.margin <= {self.margin} 
                                                            #AND exists (SELECT * FROM stock_etf ETF WHERE YES_DAY.code=ETF.code)
                                                            #AND exists (SELECT * FROM stock_kospi d WHERE YES_DAY.code=d.code)
                                        
                                    
        elif self.db_to_realtime_daily_buy_list_num == 21:
                                
                        ma_period = 70
                        realtime_daily_buy_list = []
                        realtime_daily_buy_list_1 = []
                        realtime_daily_buy_list_2 = []
                       # realtime_daily_buy_list_3 = []
                        self.date_before_l = 240
                        if i < self.date_before_l + 1:
                            
                            pass
                        
                        else:        
                            # date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            # date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            # date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            # date_before_d =self.date_rows[i - 1 - self.date_before_d][0]
    
                            # date_before_e = self.date_rows[i -1 -1][0]      
                            
                            #if i > ma_period:
                                            # 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 (11-28 추가)

                                            sql = f"""
                                                    SELECT YES_DAY.*
                                                    FROM `{date_rows_yesterday}` YES_DAY ,stock_info info 
                                                    WHERE YES_DAY.code = info.code 
                                                    and info.audit = '{self.audit}'  
                                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                                    and info.remarks NOT LIKE '{self.remarks_stop}'    
                                                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_b}', '{self.stock_market_c}', '{self.stock_market_d}')   
                                                    and NOT exists (select * from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                                    and NOT exists (select * from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                                    and NOT exists (select * from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                                    and NOT exists (select * from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                                    and NOT exists (select * from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code)  
                                                    AND 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 > YES_DAY.close
                                                    AND YES_DAY.volume != 0 
                                                    AND YES_DAY.close < {self.invest_unit}
                                                    ORDER BY YES_DAY.volume * YES_DAY.close DESC 
                                                """
                                            realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql).fetchall()
                                            
                                            # 과매도 포지션 포착
                                            for item in realtime_daily_buy_list_temp:
                                                code_name = item.code_name
                                                # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                                bb_sql = f"""
                                                        SELECT close
                                                        FROM `{code_name}`
                                                        WHERE date <= '{date_rows_yesterday}'
                                                        ORDER BY date DESC limit {ma_period}
                                                    """
                                                df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                                if len(df_close) >= ma_period:
                                                    # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                                    result = BBands(pd.DataFrame(df_close), w=ma_period)
                                                    # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                                    if result:
                                                        mbb, ubb, lbb, perb, bw = result
                                                        # perb가 음수인 경우: 현재종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                                        if perb < 0:
                                                            items_code_name = item.code_name
                                                            items = f"""
                                                                    SELECT *
                                                                    FROM `{date_rows_yesterday}` 
                                                                    WHERE  code_name = '{items_code_name}'
                                                                    """
                                                            items = self.engine_daily_buy_list.execute(items).fetchall()
                                            
                                        

                                                            # 종목코드
                                                            code = items[0][4]
                                                            #code_name = row[5]
                                                            # 어제 종가
                                                            yes_close = items[0][7]

                                                            yes_open = items[0][8]
                                                            # date_rows_yesterday 가 self.date_rows[i-1] 값이다.
                                                            # 어제 일자 기준 n 일전 날짜
                                                            #date_before = self.date_rows[i-1-self.day_before][0]
                                                            
                                                            date_before_a =self.date_rows[i - 1 - 20][0]
                                                            date_before_b =self.date_rows[i - 1 - 40][0]
                                                            date_before_c =self.date_rows[i - 1 - 60][0]
                                                            date_before_d =self.date_rows[i - 1 - 80][0]
                                                            date_before_e =self.date_rows[i - 1 - 100][0]
                                                            date_before_f =self.date_rows[i - 1 - 120][0]
                                                            date_before_g =self.date_rows[i - 1 - 140][0]
                                                            date_before_h =self.date_rows[i - 1 - 160][0]
                                                            date_before_i =self.date_rows[i - 1 - 180][0]
                                                            date_before_j =self.date_rows[i - 1 - 200][0]
                                                            date_before_k =self.date_rows[i - 1 - 220][0]
                                                            date_before_l =self.date_rows[i - 1 - 240][0]
                                                            # date_before_m =self.date_rows[i - 1 - 200][0]
                                                            # date_before_n =self.date_rows[i - 1 - 200][0]
                                                            
                                                            # 어제 일자 기준 n 일전 종가
                                                            #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                                                            date_before_close_a = self.get_now_close_price_by_date(code, date_before_a)
                                                            date_before_close_b = self.get_now_close_price_by_date(code, date_before_b)
                                                            date_before_close_c = self.get_now_close_price_by_date(code, date_before_c)
                                                            date_before_close_d = self.get_now_close_price_by_date(code, date_before_d)
                                                            date_before_close_e = self.get_now_close_price_by_date(code, date_before_e)
                                                            
                                                            date_before_close_f = self.get_now_close_price_by_date(code, date_before_f)
                                                            date_before_close_g = self.get_now_close_price_by_date(code, date_before_g)
                                                            date_before_close_h = self.get_now_close_price_by_date(code, date_before_h)
                                                            date_before_close_i = self.get_now_close_price_by_date(code, date_before_i)
                                                            date_before_close_j = self.get_now_close_price_by_date(code, date_before_j)

                                                            date_before_close_k = self.get_now_close_price_by_date(code, date_before_k)
                                                            date_before_close_l = self.get_now_close_price_by_date(code, date_before_l)
                                                        
                                                            if date_before_close_a != 0 and date_before_close_a != False and date_before_close_b != 0 and date_before_close_b != False and date_before_close_c != 0 and date_before_close_c != False and date_before_close_d != 0 and date_before_close_d != False and date_before_close_e != 0 and date_before_close_e != False and date_before_close_f != 0 and date_before_close_f != False and date_before_close_g != 0 and date_before_close_g != False and date_before_close_h != 0 and date_before_close_h != False and date_before_close_i != 0 and date_before_close_i != False and date_before_close_j != 0 and date_before_close_j != False and date_before_close_k != 0 and date_before_close_k != False and date_before_close_l != 0 and date_before_close_l != False: 
                                                                # 모멘텀 계산 : n일전 종가 대비 수익률
                                                                diff_point_calc = (((yes_close - date_before_close_a) / date_before_close_a * 100) +
                                                                                    ((yes_close - date_before_close_b) / date_before_close_b * 100) +
                                                                                    ((yes_close - date_before_close_c) / date_before_close_c * 100) + 
                                                                                    ((yes_close - date_before_close_d) / date_before_close_d * 100) +
                                                                                    ((yes_close - date_before_close_e) / date_before_close_e * 100) +
                                                                                    ((yes_close - date_before_close_f) / date_before_close_f * 100) +
                                                                                    ((yes_close - date_before_close_g) / date_before_close_g * 100) +
                                                                                    ((yes_close - date_before_close_h) / date_before_close_h * 100) +
                                                                                    ((yes_close - date_before_close_i) / date_before_close_i * 100) +
                                                                                    ((yes_close - date_before_close_j) / date_before_close_j * 100) +
                                                                                    ((yes_close - date_before_close_k) / date_before_close_k * 100) +
                                                                                    ((yes_close - date_before_close_l) / date_before_close_l * 100)) / 12


                                                                # 모멘텀(수익률)이 self.diff_point 보다 높을 경우 realtime_daily_buy_list에 append
                                                                #if ((diff_point_calc > self.diff_point) and (diff_point_calc_2 > self.diff_point)) or ((diff_point_calc_3 > self.diff_point) and (diff_point_calc_4 >self.diff_point)):                                                           
                                                                   # realtime_daily_buy_list_2.append(items)

                                                                if diff_point_calc >  self.diff_point: 
                                                                    #code_name = str(row.code_name)
                                                                    #code  =  int(row.code)                                  
                                                                    sql = f"""
                                                                        SELECT *
                                                                        FROM `{date_rows_yesterday}`                                                                    
                                                                        where code = {code}
                                                                        """
                                                                    realtime_daily_buy_list += self.engine_daily_buy_list.execute(sql).fetchall()
                                                                    # realtime_daily_buy_list.append(realtime_daily_buy_list_2)    
        elif self.db_to_realtime_daily_buy_list_num == 22:
                                
                        ma_period = 70 
                        realtime_daily_buy_list = []
                        realtime_daily_buy_list_1 = []
                        realtime_daily_buy_list_2 = []
                        date_before_e =self.date_rows[i - 1 - 1][0]
                       # realtime_daily_buy_list_3 = []
                        self.date_before_l = 240
                        if i < self.date_before_l + 1:
                            
                            pass
                        
                        else:        
                            # date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            # date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            # date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            # date_before_d =self.date_rows[i - 1 - self.date_before_d][0]
    
                            # date_before_e = self.date_rows[i -1 -1][0]      
                            
                            #if i > ma_period:
                                            # 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 (11-28 추가)

                                            sql = f"""
                                                    SELECT YES_DAY.*
                                                    FROM `{date_rows_yesterday}` YES_DAY ,stock_info info  
                                                    WHERE YES_DAY.code = info.code 
                                                    and info.audit = '{self.audit}'  
                                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                                    and info.remarks NOT LIKE '{self.remarks_stop}'  
                                                    and info.stock_market IN ('{self.stock_market_a}','{self.stock_market_b}','{self.stock_market_c}')   
                                                    and NOT exists (select * from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                                    and NOT exists (select * from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                                    and NOT exists (select * from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                                    and NOT exists (select * from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                                    and NOT exists (select * from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code)  
                                                    AND YES_DAY.volume != 0 
                                                    AND YES_DAY.close < {self.invest_unit}
                                                    AND 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 > YES_DAY.close
                                                    ORDER BY YES_DAY.volume * YES_DAY.close DESC 
                                                """
                                            realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql).fetchall()

                                            # 과매도 포지션 포착
                                            for item in realtime_daily_buy_list_temp:
                                                code_name = item.code_name
                                                # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                                bb_sql = f"""
                                                        SELECT close
                                                        FROM `{code_name}`
                                                        WHERE date <= '{date_rows_yesterday}'
                                                        ORDER BY date DESC limit {ma_period}
                                                    """
                                                df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                                if len(df_close) >= ma_period:
                                                    # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                                    result = BBands(pd.DataFrame(df_close), w=ma_period)
                                                    # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                                    if result:
                                                        mbb, ubb, lbb, perb, bw = result
                                                        # perb가 음수인 경우: 종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                                        if perb < 0:
                                                            items_code_name = item.code_name
                                                            items = f"""
                                                                    SELECT *
                                                                    FROM `{date_rows_yesterday}` 
                                                                    WHERE  code_name = '{items_code_name}'
                                                                    """
                                                            items = self.engine_daily_buy_list.execute(items).fetchall()
                                            
                                        

                                                            # 종목코드
                                                            code = items[0][4]
                                                            code_name = items[0][5]
                                                            #code_name = row[5]
                                                            # 어제 종가
                                                            yes_close = items[0][7]

                                                            yes_open = items[0][8]
                                                            # date_rows_yesterday 가 self.date_rows[i-1] 값이다.
                                                            # 어제 일자 기준 n 일전 날짜
                                                            #date_before = self.date_rows[i-1-self.day_before][0]
                                                            
                                                            date_before_a =self.date_rows[i - 1 - 20][0]
                                                            date_before_b =self.date_rows[i - 1 - 40][0]
                                                            date_before_c =self.date_rows[i - 1 - 60][0]
                                                            date_before_d =self.date_rows[i - 1 - 80][0]
                                                            date_before_e =self.date_rows[i - 1 - 100][0]
                                                            date_before_f =self.date_rows[i - 1 - 120][0]
                                                            date_before_g =self.date_rows[i - 1 - 140][0]
                                                            date_before_h =self.date_rows[i - 1 - 160][0]
                                                            date_before_i =self.date_rows[i - 1 - 180][0]
                                                            date_before_j =self.date_rows[i - 1 - 200][0]
                                                            date_before_k =self.date_rows[i - 1 - 220][0]
                                                            date_before_l =self.date_rows[i - 1 - 240][0]
                                                            
                                                            date_before_m =self.date_rows[i - 1 - 14][0]
                                                            date_before =self.date_rows[i -1][0]
                                                            # date_before_n =self.date_rows[i - 1 - 200][0]
                                                            
                                                            # 어제 일자 기준 n 일전 종가
                                                            #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                                                            date_before_close_a = self.get_now_close_price_by_date(code, date_before_a)
                                                            date_before_close_b = self.get_now_close_price_by_date(code, date_before_b)
                                                            date_before_close_c = self.get_now_close_price_by_date(code, date_before_c)
                                                            date_before_close_d = self.get_now_close_price_by_date(code, date_before_d)
                                                            date_before_close_e = self.get_now_close_price_by_date(code, date_before_e)
                                                            
                                                            date_before_close_f = self.get_now_close_price_by_date(code, date_before_f)
                                                            date_before_close_g = self.get_now_close_price_by_date(code, date_before_g)
                                                            date_before_close_h = self.get_now_close_price_by_date(code, date_before_h)
                                                            date_before_close_i = self.get_now_close_price_by_date(code, date_before_i)
                                                            date_before_close_j = self.get_now_close_price_by_date(code, date_before_j)

                                                            date_before_close_k = self.get_now_close_price_by_date(code, date_before_k)
                                                            date_before_close_l = self.get_now_close_price_by_date(code, date_before_l)
                                                                                         
                                                            #date_before_close_m = self.get_now_rsi_price_by_date(code_name, date_before_m)
                                                            if date_before_close_a != 0 and date_before_close_a != False and date_before_close_b != 0 and date_before_close_b != False and date_before_close_c != 0 and date_before_close_c != False and date_before_close_d != 0 and date_before_close_d != False and date_before_close_e != 0 and date_before_close_e != False and date_before_close_f != 0 and date_before_close_f != False and date_before_close_g != 0 and date_before_close_g != False and date_before_close_h != 0 and date_before_close_h != False and date_before_close_i != 0 and date_before_close_i != False and date_before_close_j != 0 and date_before_close_j != False and date_before_close_k != 0 and date_before_close_k != False and date_before_close_l != 0 and date_before_close_l != False: 
                                                                # 모멘텀 계산 : n일전 종가 대비 수익률
                                                                diff_point_calc = (((yes_close - date_before_close_a) / date_before_close_a * 100) +
                                                                                    ((yes_close - date_before_close_b) / date_before_close_b * 100) +
                                                                                    ((yes_close - date_before_close_c) / date_before_close_c * 100) + 
                                                                                    ((yes_close - date_before_close_d) / date_before_close_d * 100) +
                                                                                    ((yes_close - date_before_close_e) / date_before_close_e * 100) +
                                                                                    ((yes_close - date_before_close_f) / date_before_close_f * 100) +
                                                                                    ((yes_close - date_before_close_g) / date_before_close_g * 100) +
                                                                                    ((yes_close - date_before_close_h) / date_before_close_h * 100) +
                                                                                    ((yes_close - date_before_close_i) / date_before_close_i * 100) +
                                                                                    ((yes_close - date_before_close_j) / date_before_close_j * 100) +
                                                                                    ((yes_close - date_before_close_k) / date_before_close_k * 100) +
                                                                                    ((yes_close - date_before_close_l) / date_before_close_l * 100)) / 12


                                                                # 모멘텀(수익률)이 self.diff_point 보다 높을 경우 realtime_daily_buy_list에 append
                                                                #if ((diff_point_calc > self.diff_point) and (diff_point_calc_2 > self.diff_point)) or ((diff_point_calc_3 > self.diff_point) and (diff_point_calc_4 >self.diff_point)):                                                           
                                                                   # realtime_daily_buy_list_2.append(items)
                                                                sql = "select close from `" + code_name + "` where date >= '%s' and date <= '%s' ORDER BY date desc"
                                                                date_before_close_m = self.engine_daily_craw.execute(sql%(date_before_m, date_before)).fetchall()
                                                                 # rsi 참고자료        
                                                                    # https://wikidocs.net/163550#rsi   
                                                                # variable = date_before_close_m[0][0] - date_before_close_m[1][0]
                                                                # #variable_abs = variable
                                                                # Up_a  = np.where(variable>=0, variable, 0)
                                                                # #Up_a_DataFrame  = pd.DataFrame(Up_a)
                                                                # down_b = np.where(variable < 0, abs(variable), 0)
                                                                # #down_b_DataFrame  = pd.DataFrame(down_b)   
                                                                # AU = Up_a.ewm(alpha=1/14, min_periods=14).mean()
                                                                # AD = down_b.ewm(alpha=1/14, min_periods=14).mean()
                                                                #df['RS'] = df['AU'] / df['AD']
                                                                #df['RSI'] = 100 - (100 / (1 + df['RS']))
                                                                # AU = pd.DataFrame(date_before_close_m).rolling(window=14).mean()
                                                                # AD = pd.DataFrame(date_before_close_m).rolling(window=14).mean()
                                                               
                                                                # RSI = AU / (AD + AU) * 100
                                                                #############################3
                                                                # 참고 사이트 -https://hotorch.tistory.com/366
                                                                date_before_close_m = pd.DataFrame(date_before_close_m[:])
                                                                # delta = date_before_close_m.diff()
                                                                # up ,down = delta.copy(),delta.copy()
                                                                # up[up<0] = 0
                                                                # down[down >0] = 0
                                                                # _gain = pd.DataFrame(date_before_close_m).rolling(window=14).mean()
                                                                # _loss = pd.DataFrame(down).rolling(window=14).mean()
                                                                # RS = _gain / _loss
                                                                # rsi_14 = pd.DataFrame(100 -(100 / (1+ RS)), name="RSI") 
                                                                # ######
                                                                #data[0] = RSI(date_before_close_m, period = 14)
    


                                                                if diff_point_calc >  self.diff_point: 
                                                                    
                                                                    #code_name = str(row.code_name)
                                                                    #code  =  int(row.code)                                  
                                                                    sql = f"""
                                                                        SELECT *
                                                                        FROM `{date_rows_yesterday}`                                                                    
                                                                        where code = {code}
                                                                        """
                                                                    realtime_daily_buy_list += self.engine_daily_buy_list.execute(sql).fetchall()                               
        elif self.db_to_realtime_daily_buy_list_num == 23:
   
                        ma_period = 20
                        realtime_daily_buy_list = []
                        realtime_daily_buy_list_1 = []
                        realtime_daily_buy_list_2 = []
                       # realtime_daily_buy_list_3 = []
                        self.date_before_l = 240
                        if i < self.date_before_l + 1:
                            
                            pass
                        
                        else:        
    
                            sql = f"""
                                    SELECT YES_DAY.*
                                    FROM `{date_rows_yesterday}` YES_DAY ,stock_info info 
                                    WHERE YES_DAY.code = info.code 
                                    and info.audit = '{self.audit}'  
                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                    and info.remarks NOT LIKE '{self.remarks_stop}'    
                                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_b}', '{self.stock_market_c}')   
                                    and NOT exists (select * from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                    and NOT exists (select * from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                    and NOT exists (select * from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                    and NOT exists (select * from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                    and NOT exists (select * from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code)  
                                    AND 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 > YES_DAY.close
                                    ORDER BY YES_DAY.volume * YES_DAY.close DESC 
                                """
                            realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql).fetchall()
        
                       
                        
                            # 과매도 포지션 포착
                            for item in realtime_daily_buy_list_temp:
                                code_name = item.code_name
                                # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                bb_sql = f"""
                                        SELECT close
                                        FROM `{code_name}`
                                        WHERE date <= '{date_rows_yesterday}'
                                        ORDER BY date DESC limit {ma_period}
                                    """
                                df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                if len(df_close) >= ma_period:
                                    # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                    result = BBands(pd.DataFrame(df_close), w=ma_period)
                                    # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                    if result:
                                        mbb, ubb, lbb, perb, bw = result
                                        # perb가 음수인 경우: 현재종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                        if perb < 0:
                                            items_code_name = item.code_name
                                            items = f"""
                                                    SELECT *
                                                    FROM `{date_rows_yesterday}` 
                                                    WHERE  code_name = '{items_code_name}'
                                                    """
                                            items = self.engine_daily_buy_list.execute(items).fetchall()
                            
                        

                                            # 종목코드
                                            code = items[0][4]
                                            #code_name = row[5]
                                            # 어제 종가
                                            yes_close = items[0][7]

                                            yes_open = items[0][8]
                                            # date_rows_yesterday 가 self.date_rows[i-1] 값이다.
                                            # 어제 일자 기준 n 일전 날짜
                                            #date_before = self.date_rows[i-1-self.day_before][0]
                                            
                                            date_before_a =self.date_rows[i - 1 - 20][0]
                                            date_before_b =self.date_rows[i - 1 - 40][0]
                                            date_before_c =self.date_rows[i - 1 - 60][0]
                                            date_before_d =self.date_rows[i - 1 - 80][0]
                                            date_before_e =self.date_rows[i - 1 - 100][0]
                                            date_before_f =self.date_rows[i - 1 - 120][0]
                                            date_before_g =self.date_rows[i - 1 - 140][0]
                                            date_before_h =self.date_rows[i - 1 - 160][0]
                                            date_before_i =self.date_rows[i - 1 - 180][0]
                                            date_before_j =self.date_rows[i - 1 - 200][0]
                                            date_before_k =self.date_rows[i - 1 - 220][0]
                                            date_before_l =self.date_rows[i - 1 - 240][0]
                                            # date_before_m =self.date_rows[i - 1 - 200][0]
                                            # date_before_n =self.date_rows[i - 1 - 200][0]
                                            
                                            # 어제 일자 기준 n 일전 종가
                                            #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                                            date_before_close_a = self.get_now_clo120_price_by_date(code, date_before_f)
                                            # date_before_close_b = self.get_now_clo5_price_by_date(code, date_before_b)
                                            # date_before_close_c = self.get_now_clo5_price_by_date(code, date_before_c)
                                            # date_before_close_d = self.get_now_clo5_price_by_date(code, date_before_d)
                                            # date_before_close_e = self.get_now_clo5_price_by_date(code, date_before_e)
                                            
                                            # date_before_close_f = self.get_now_clo5_price_by_date(code, date_before_f)
                                            # date_before_close_g = self.get_now_clo5_price_by_date(code, date_before_g)
                                            # date_before_close_h = self.get_now_clo5_price_by_date(code, date_before_h)
                                            # date_before_close_i = self.get_now_clo5_price_by_date(code, date_before_i)
                                            # date_before_close_j = self.get_now_clo5_price_by_date(code, date_before_j)

                                            # date_before_close_k = self.get_now_clo5_price_by_date(code, date_before_k)
                                            # date_before_close_l = self.get_now_clo5_price_by_date(code, date_before_l)
                                        
                                            if date_before_close_a != 0 and date_before_close_a != False: 
                                                # 모멘텀 계산 : n일전 종가 대비 수익률
                                                diff_point_calc = ((yes_close - date_before_close_a) / date_before_close_a * 100)
                                                                    


                                                # 모멘텀(수익률)이 self.diff_point 보다 높을 경우 realtime_daily_buy_list에 append
                                                #if ((diff_point_calc > self.diff_point) and (diff_point_calc_2 > self.diff_point)) or ((diff_point_calc_3 > self.diff_point) and (diff_point_calc_4 >self.diff_point)):                                                           
                                                    # realtime_daily_buy_list_2.append(items)

                                                if diff_point_calc >  self.diff_point: 
                                                    #code_name = str(row.code_name)
                                                    #code  =  int(row.code)                                  
                                                    sql = f"""
                                                        SELECT *
                                                        FROM `{date_rows_yesterday}`                                                                    
                                                        where code = {code}
                                                        """
                                                    realtime_daily_buy_list += self.engine_daily_buy_list.execute(sql).fetchall()
                                                # realtime_daily_buy_list.append(realtime_daily_buy_list_2)    
        elif self.db_to_realtime_daily_buy_list_num == 24:
                                
                        ma_period = 20
                        realtime_daily_buy_list = []
                        realtime_daily_buy_list_1 = []
                        realtime_daily_buy_list_2 = []
                       # realtime_daily_buy_list_3 = []
                        self.date_before_l = 240
                        if i < self.date_before_l + 1:
                            
                            pass
                        
                        else:        
                            # date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            # date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            # date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            # date_before_d =self.date_rows[i - 1 - self.date_before_d][0]
    
                            # date_before_e = self.date_rows[i -1 -1][0]      
                            
                            #if i > ma_period:
                                            # 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 (11-28 추가)
                                            #AND YES_DAY.volume != 0 
                                            sql = f"""
                                                    SELECT YES_DAY.*
                                                    FROM `{date_rows_yesterday}` YES_DAY ,stock_info info 
                                                    WHERE YES_DAY.code = info.code 
                                                    and info.audit = '{self.audit}'  
                                                    and info.remarks NOT LIKE '{self.remarks_manage}'  
                                                    and info.remarks NOT LIKE '{self.remarks_stop}'    
                                                    and info.stock_market IN ('{self.stock_market_a}', '{self.stock_market_b}', '{self.stock_market_c}')   
                                                    and NOT exists (select * from stock_managing c where YES_DAY.code=c.code and c.code_name != '' group by c.code) 
                                                    and NOT exists (select * from stock_insincerity d where YES_DAY.code=d.code and d.code_name !='' group by d.code) 
                                                    and NOT exists (select * from stock_invest_caution e where YES_DAY.code=e.code and DATE_SUB({date_rows_yesterday}, INTERVAL {self.interval_month} MONTH ) < e.post_date and e.post_date < Date({date_rows_yesterday}) and e.type != '투자경고 지정해제' group by e.code) 
                                                    and NOT exists (select * from stock_invest_warning f where YES_DAY.code=f.code and f.post_date <= DATE({date_rows_yesterday}) and (f.cleared_date > DATE({date_rows_yesterday}) or f.cleared_date is null) group by f.code) 
                                                    and NOT exists (select * from stock_invest_danger g where YES_DAY.code=g.code and g.post_date <= DATE({date_rows_yesterday}) and (g.cleared_date > DATE({date_rows_yesterday}) or g.cleared_date is null) group by g.code)  
                                                    AND 0.25 * YES_DAY.clo5 + 0.25 * YES_DAY.clo10 + 0.25 * YES_DAY.clo20 + 0.25 * YES_DAY.clo40 > YES_DAY.close
                                                    ORDER BY YES_DAY.volume * YES_DAY.close DESC 
                                                """
                                            realtime_daily_buy_list_temp = self.engine_daily_buy_list.execute(sql).fetchall()
                                            
                                            # 과매도 포지션 포착
                                            for item in realtime_daily_buy_list_temp:
                                                code_name = item.code_name
                                                # 위의 조건을 충족하는 종목의 종가 데이터들을 가져오는 쿼리
                                                bb_sql = f"""
                                                        SELECT close
                                                        FROM `{code_name}`
                                                        WHERE date <= '{date_rows_yesterday}'
                                                        ORDER BY date DESC limit {ma_period}
                                                    """
                                                df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                                                if len(df_close) >= ma_period:
                                                    # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                                    result = BBands(pd.DataFrame(df_close), w=ma_period)
                                                    # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                                    if result:
                                                        mbb, ubb, lbb, perb, bw = result
                                                        # perb가 음수인 경우: 전날종가가 볼린저밴드 하한선보다 아래에 위치 할 경우 매수리스트에 넣는다
                                                        if perb < 0:
                                                            items_code_name = item.code_name
                                                            items = f"""
                                                                    SELECT *
                                                                    FROM `{date_rows_yesterday}` 
                                                                    WHERE  code_name = '{items_code_name}'
                                                                    """
                                                            items = self.engine_daily_buy_list.execute(items).fetchall()
                                            
                                        

                                                            # 종목코드
                                                            code = items[0][4]
                                                            #code_name = row[5]
                                                            # 어제 종가
                                                            yes_close = items[0][7]

                                                            yes_open = items[0][8]
                                                            # date_rows_yesterday 가 self.date_rows[i-1] 값이다.
                                                            # 어제 일자 기준 n 일전 날짜
                                                            #date_before = self.date_rows[i-1-self.day_before][0]
                                                            
                                                            date_before_a =self.date_rows[i - 1 - 20][0]
                                                            date_before_b =self.date_rows[i - 1 - 40][0]
                                                            date_before_c =self.date_rows[i - 1 - 60][0]
                                                            date_before_d =self.date_rows[i - 1 - 80][0]
                                                            date_before_e =self.date_rows[i - 1 - 100][0]
                                                            date_before_f =self.date_rows[i - 1 - 120][0]
                                                            date_before_g =self.date_rows[i - 1 - 140][0]
                                                            date_before_h =self.date_rows[i - 1 - 160][0]
                                                            date_before_i =self.date_rows[i - 1 - 180][0]
                                                            date_before_j =self.date_rows[i - 1 - 200][0]
                                                            date_before_k =self.date_rows[i - 1 - 220][0]
                                                            date_before_l =self.date_rows[i - 1 - 240][0]
                                                            # date_before_m =self.date_rows[i - 1 - 200][0]
                                                            # date_before_n =self.date_rows[i - 1 - 200][0]
                                                            
                                                            # 어제 일자 기준 n 일전 종가
                                                            #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                                                            date_before_close_a = self.get_now_clo20_price_by_date(code, date_before_a)
                                                            date_before_close_b = self.get_now_clo20_price_by_date(code, date_before_b)
                                                            date_before_close_c = self.get_now_clo20_price_by_date(code, date_before_c)
                                                            date_before_close_d = self.get_now_clo20_price_by_date(code, date_before_d)
                                                            date_before_close_e = self.get_now_clo20_price_by_date(code, date_before_e)
                                                            
                                                            date_before_close_f = self.get_now_clo20_price_by_date(code, date_before_f)
                                                            date_before_close_g = self.get_now_clo20_price_by_date(code, date_before_g)
                                                            date_before_close_h = self.get_now_clo20_price_by_date(code, date_before_h)
                                                            date_before_close_i = self.get_now_clo20_price_by_date(code, date_before_i)
                                                            date_before_close_j = self.get_now_clo20_price_by_date(code, date_before_j)

                                                            date_before_close_k = self.get_now_clo20_price_by_date(code, date_before_k)
                                                            date_before_close_l = self.get_now_clo20_price_by_date(code, date_before_l)
                                                        
                                                            if date_before_close_a != 0 and date_before_close_a != False and date_before_close_b != 0 and date_before_close_b != False and date_before_close_c != 0 and date_before_close_c != False and date_before_close_d != 0 and date_before_close_d != False and date_before_close_e != 0 and date_before_close_e != False and date_before_close_f != 0 and date_before_close_f != False and date_before_close_g != 0 and date_before_close_g != False and date_before_close_h != 0 and date_before_close_h != False and date_before_close_i != 0 and date_before_close_i != False and date_before_close_j != 0 and date_before_close_j != False and date_before_close_k != 0 and date_before_close_k != False and date_before_close_l != 0 and date_before_close_l != False: 
                                                                # 모멘텀 계산 : n일전 종가 대비 수익률
                                                                diff_point_calc = (((yes_close - date_before_close_a) / date_before_close_a * 100) +
                                                                                    ((yes_close - date_before_close_b) / date_before_close_b * 100) +
                                                                                    ((yes_close - date_before_close_c) / date_before_close_c * 100) + 
                                                                                    ((yes_close - date_before_close_d) / date_before_close_d * 100) +
                                                                                    ((yes_close - date_before_close_e) / date_before_close_e * 100) +
                                                                                    ((yes_close - date_before_close_f) / date_before_close_f * 100) +
                                                                                    ((yes_close - date_before_close_g) / date_before_close_g * 100) +
                                                                                    ((yes_close - date_before_close_h) / date_before_close_h * 100) +
                                                                                    ((yes_close - date_before_close_i) / date_before_close_i * 100) +
                                                                                    ((yes_close - date_before_close_j) / date_before_close_j * 100) +
                                                                                    ((yes_close - date_before_close_k) / date_before_close_k * 100) +
                                                                                    ((yes_close - date_before_close_l) / date_before_close_l * 100)) / 12


                                                                # 모멘텀(수익률)이 self.diff_point 보다 높을 경우 realtime_daily_buy_list에 append
                                                                #if ((diff_point_calc > self.diff_point) and (diff_point_calc_2 > self.diff_point)) or ((diff_point_calc_3 > self.diff_point) and (diff_point_calc_4 >self.diff_point)):                                                           
                                                                   # realtime_daily_buy_list_2.append(items)

                                                                if diff_point_calc >  self.diff_point: 
                                                                    #code_name = str(row.code_name)
                                                                    #code  =  int(row.code)                                  
                                                                    sql = f"""
                                                                        SELECT *
                                                                        FROM `{date_rows_yesterday}`                                                                    
                                                                        where code = {code}
                                                                        """
                                                                    realtime_daily_buy_list += self.engine_daily_buy_list.execute(sql).fetchall()
                                                                    # realtime_daily_buy_list.append(realtime_daily_buy_list_2)    



        ######################################################################################################################################################################################
        else:
            print(f"{self.simul_num}번 알고리즘에 대한 self.db_to_realtime_daily_buy_list_num 설정이 비었습니다. variable_setting 함수에서 self.db_to_realtime_daily_buy_list_num 을 확인해주세요.")
            sys.exit(1)
        # realtime_daily_buy_list 에 종목이 하나라도 있다면, 즉 매수할 종목이 하나라도 있다면 아래 로직을 들어간다.
        if len(realtime_daily_buy_list) > 0:
            # realtime_daily_buy_list 라는 리스트를 df_realtime_daily_buy_list 라는 데이터프레임으로 변환하는 과정
            # 차이점은 리스트는 컬럼에 대한 개념이 없는데, 데이터프레임은 컬럼이 있다.

            df_realtime_daily_buy_list = DataFrame(realtime_daily_buy_list,
                                                   columns=['index', 'index2', 'date', 'check_item', 'code',
                                                            'code_name', 'd1_diff_rate', 'close', 'open', 'high',
                                                            'low', 'volume',
                                                            'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
                                                            'clo100', 'clo120',
                                                            "clo5_diff_rate", "clo10_diff_rate", "clo20_diff_rate",
                                                            "clo40_diff_rate", "clo60_diff_rate", "clo80_diff_rate",
                                                            "clo100_diff_rate", "clo120_diff_rate",
                                                            'yes_clo5', 'yes_clo10', 'yes_clo20', 'yes_clo40',
                                                            'yes_clo60',
                                                            'yes_clo80',
                                                            'yes_clo100', 'yes_clo120',
                                                            'vol5', 'vol10', 'vol20', 'vol40', 'vol60', 'vol80',
                                                            'vol100', 'vol120'])

            # lamda는 익명 함수이다. 여기서 int로 param을 보내야 6d ( 정수) 에서 안걸린다.
            df_realtime_daily_buy_list['code'] = df_realtime_daily_buy_list['code'].apply(
                lambda x: "{:0>6d}".format(int(x)))

            # 시뮬레이터의 경우
            if self.op != 'real':
                df_realtime_daily_buy_list['check_item'] = int(0)
                # [to_sql]
                # df_realtime_daily_buy_list 라는 데이터프레임을
                # simulator 데이터베이스의 realtime_daily_buy_list 테이블로 만들어주는 명령
                #
                # ** if_exists 옵션 **
                # # 데이터베이스에 테이블이 존재할 때 수행 동작을 지정한다.
                # 'fail', 'replace', 'append' 중 하나를 사용할 수 있는데 기본값은 'fail'이다.
                # 'fail'은 데이터베이스에 테이블이 있다면 아무 동작도 수행하지 않는다.
                # 'replace'는 테이블이 존재하면 기존 테이블을 삭제하고 새로 테이블을 생성한 후 데이터를 삽입한다.
                # 'append'는 테이블이 존재하면 데이터만을 추가한다.
                df_realtime_daily_buy_list.to_sql('realtime_daily_buy_list', self.engine_simulator, if_exists='replace')

                # 현재 보유 중인 종목은 매수 리스트(realtime_daily_buy_list) 에서 제거 하는 로직
                if self.is_simul_table_exist(self.db_name, "all_item_db"):
                    sql = "delete from realtime_daily_buy_list where code in (select code from all_item_db where sell_date = '%s' or buy_date = '%s' or sell_date = '%s')"
                    # delete는 리턴 값이 없기 때문에 fetchall 쓰지 않는다.
                    self.engine_simulator.execute(sql % (0, date_rows_today, date_rows_today))

                # AI 부분.
                if self.use_ai:
                    from ai_filter import ai_filter
                    ai_filter(self.ai_filter_num, engine=self.engine_simulator, until=date_rows_yesterday)

                # 최종적으로 realtime_daily_buy_list 테이블에 저장 된 종목들을 가져온다.
                self.get_realtime_daily_buy_list()

            # 모의, 실전 투자 봇 의 경우
            else:
                # check_item 컬럼에 0 으로 setting
                df_realtime_daily_buy_list['check_item'] = int(0)
                df_realtime_daily_buy_list.to_sql('realtime_daily_buy_list', self.engine_simulator, if_exists='replace')

                # 현재 보유 중인 종목들은 삭제
                sql = "delete from realtime_daily_buy_list where code in (select code from possessed_item)"
                self.engine_simulator.execute(sql)


        # 매수할 종목이 없으면, df_realtime_daily_buy_list라는 데이터프레임의 길이를 저장하는
        # len_df_realtime_daily_buy_list에 다가 0을 넣는다.
        else:
            self.len_df_realtime_daily_buy_list = 0
            #추가 코드 (매수 조건에 맞는 종목이 하나도 없을 경우 realtime_daily_buy_list 를 비워준다)
            if self.engine_simulator.dialect.has_table(self.engine_simulator, "realtime_daily_buy_list"):
                self.engine_simulator.execute("""
                    DELETE FROM realtime_daily_buy_list 
                """)

    # 현재의 주가를 all_item_db에 있는 보유한 종목들에 대해서 반영 한다.
    def db_to_all_item_present_price_update(self, code_name, d1_diff_rate, close, open, high, low, volume, clo5, clo10, clo20,
                                                         clo40, clo60, clo80, clo100, clo120, option='ALL'):
        
        if self.op == 'real': # 콜렉터에서 업데이트 할 때는 현재가를 종가로 업데이트(trader에서 실시간으로 present_price 업데이트함)
            present_price = close
        else:
            present_price = open # 시뮬레이터에서는 open가를 현재가로 업데이트

        # option이 ALL이면 모든 데이터 업데이트
        if option == "ALL":
            sql = f"update all_item_db set d1_diff_rate = {d1_diff_rate}, close = {close}, open = {open}, high = {high}, " \
                  f"low = {low}, volume = {volume}, present_price = {present_price}, clo5 = {clo5}, clo10 = {clo10}, clo20 = {clo20}, " \
                  f"clo40 = {clo40}, clo60 = {clo60}, clo80 = {clo80}, clo100 = {clo100}, clo120 = {clo120} " \
                  f"where code_name = '{code_name}' and sell_date = {0}"
        # option이 OPEN이면 open, present_price 만 업데이트
        else:
            sql = f"update all_item_db set open = {open}, present_price = {present_price} where code_name = '{code_name}' and sell_date = {0}"

        self.engine_simulator.execute(sql)

    # jango_data 라는 테이블을 만들기 위한 self.jango 데이터프레임을 생성
    def init_df_jango(self):
        jango_temp = {'id': []}

        self.jango = DataFrame(jango_temp,
                               columns=['date', 'today_earning_rate', 'sum_valuation_profit', 'total_profit',
                                        'today_profit',
                                        'today_profitcut_count', 'today_losscut_count', 'today_profitcut',
                                        'today_losscut',
                                        'd2_deposit', 'total_possess_count', 'today_buy_count', 'today_buy_list_count',
                                        'today_reinvest_count',
                                        'today_cant_reinvest_count',
                                        'total_asset',
                                        'total_invest',
                                        'sum_item_total_purchase', 'total_evaluation', 'today_rate',
                                        'today_invest_price', 'today_reinvest_price',
                                        'today_sell_price', 'volume_limit', 'reinvest_point', 'sell_point',
                                        'max_reinvest_count', 'invest_limit_rate', 'invest_unit',
                                        'rate_std_sell_point', 'limit_money', 'total_profitcut', 'total_losscut',
                                        'total_profitcut_count',
                                        'total_losscut_count', 'loan_money', 'start_kospi_point',
                                        'start_kosdaq_point', 'end_kospi_point', 'end_kosdaq_point',
                                        'today_buy_total_sell_count',
                                        'today_buy_total_possess_count', 'today_buy_today_profitcut_count',
                                        'today_buy_today_profitcut_rate', 'today_buy_today_losscut_count',
                                        'today_buy_today_losscut_rate',
                                        'today_buy_total_profitcut_count', 'today_buy_total_profitcut_rate',
                                        'today_buy_total_losscut_count', 'today_buy_total_losscut_rate',
                                        'today_buy_reinvest_count0_sell_count',
                                        'today_buy_reinvest_count1_sell_count', 'today_buy_reinvest_count2_sell_count',
                                        'today_buy_reinvest_count3_sell_count', 'today_buy_reinvest_count4_sell_count',
                                        'today_buy_reinvest_count4_sell_profitcut_count',
                                        'today_buy_reinvest_count4_sell_losscut_count',
                                        'today_buy_reinvest_count5_sell_count',
                                        'today_buy_reinvest_count5_sell_profitcut_count',
                                        'today_buy_reinvest_count5_sell_losscut_count',
                                        'today_buy_reinvest_count0_remain_count',
                                        'today_buy_reinvest_count1_remain_count',
                                        'today_buy_reinvest_count2_remain_count',
                                        'today_buy_reinvest_count3_remain_count',
                                        'today_buy_reinvest_count4_remain_count',
                                        'today_buy_reinvest_count5_remain_count'],
                               index=jango_temp['id'])

    # all_item_db 라는 테이블을 만들기 위한 self.df_all_item 데이터프레임
    def init_df_all_item(self):
        df_all_item_temp = {'id': []}

        self.df_all_item = DataFrame(df_all_item_temp,
                                     columns=['id', 'order_num', 'code', 'code_name', 'rate', 'purchase_rate',
                                              'purchase_price',
                                              'present_price', 'valuation_price',
                                              'valuation_profit', 'holding_amount', 'buy_date', 'item_total_purchase',
                                              'chegyul_check', 'reinvest_count', 'reinvest_date', 'invest_unit',
                                              'reinvest_unit',
                                              'sell_date', 'sell_price', 'sell_rate', 'rate_std', 'rate_std_mod_val',
                                              'rate_std_htr', 'rate_htr',
                                              'rate_std_mod_val_htr', 'yes_close', 'close', 'd1_diff_rate', 'd1_diff',
                                              'open', 'high',
                                              'low',
                                              'volume', 'clo5', 'clo10', 'clo20', 'clo40', 'clo60', 'clo80',
                                              'clo100', 'clo120', "clo5_diff_rate", "clo10_diff_rate",
                                              "clo20_diff_rate", "clo40_diff_rate", "clo60_diff_rate",
                                              "clo80_diff_rate", "clo100_diff_rate", "clo120_diff_rate"])

    # 가장 초기에 매수 했을 때 all_item_db 에 추가하는 함수
    def db_to_all_item(self, min_date, df, index, code, code_name, purchase_price, yesterday_close):
        self.df_all_item.loc[0, 'code'] = code
        self.df_all_item.loc[0, 'code_name'] = code_name
        # 초기는 반드시 rate가 -0.33 이여야한다. -> 수수료, 세금을 반영함
        self.df_all_item.loc[0, 'rate'] = float(-0.33)

        if yesterday_close:
            self.df_all_item.loc[0, 'purchase_rate'] = round(
                (float(purchase_price) - float(yesterday_close)) / float(yesterday_close) * 100, 2)

        self.df_all_item.loc[0, 'purchase_price'] = purchase_price
        self.df_all_item.loc[0, 'present_price'] = purchase_price

        # ("code_name: "+ code_name + "purchase_price: "+ str(purchase_price))
        self.df_all_item.loc[0, 'holding_amount'] = int(self.invest_unit / purchase_price)
        self.df_all_item.loc[0, 'buy_date'] = min_date
        self.df_all_item.loc[0, 'item_total_purchase'] = self.df_all_item.loc[0, 'purchase_price'] * \
                                                         self.df_all_item.loc[
                                                             0, 'holding_amount']

        # 실시간으로 오늘 투자한 금액 합산
        self.today_invest_price = self.today_invest_price + self.df_all_item.loc[0, 'item_total_purchase']

        self.df_all_item.loc[0, 'chegyul_check'] = 0
        self.df_all_item.loc[0, 'id'] = 0
        # int로 넣어야 나중에 ++ 할수 있다.
        # self.df_all_item.loc[0, 'reinvest_date'] = '#'
        # self.df_all_item.loc[0, 'reinvest_count'] = int(0)
        # 다음에 투자할 금액은 invest_unit과 같은 금액이다.
        self.df_all_item.loc[0, 'invest_unit'] = self.invest_unit
        # self.df_all_item.loc[0, 'reinvest_unit'] = self.invest_unit
        self.df_all_item.loc[0, 'sell_rate'] = float(0)
        self.df_all_item.loc[0, 'yes_close'] = yesterday_close
        self.df_all_item.loc[0, 'close'] = df.loc[index, 'close']

        self.df_all_item.loc[0, 'open'] = df.loc[index, 'open']
        self.df_all_item.loc[0, 'high'] = df.loc[index, 'high']
        self.df_all_item.loc[0, 'low'] = df.loc[index, 'low']
        self.df_all_item.loc[0, 'volume'] = df.loc[index, 'volume']
        if df.loc[index, 'd1_diff_rate'] is not None:
            self.df_all_item.loc[0, 'd1_diff_rate'] = float(df.loc[index, 'd1_diff_rate'])
        self.df_all_item.loc[0, 'clo5'] = df.loc[index, 'clo5']
        self.df_all_item.loc[0, 'clo10'] = df.loc[index, 'clo10']
        self.df_all_item.loc[0, 'clo20'] = df.loc[index, 'clo20']
        self.df_all_item.loc[0, 'clo40'] = df.loc[index, 'clo40']
        self.df_all_item.loc[0, 'clo60'] = df.loc[index, 'clo60']
        self.df_all_item.loc[0, 'clo80'] = df.loc[index, 'clo80']
        self.df_all_item.loc[0, 'clo100'] = df.loc[index, 'clo100']
        self.df_all_item.loc[0, 'clo120'] = df.loc[index, 'clo120']

        if df.loc[index, 'clo5_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo5_diff_rate'] = float(df.loc[index, 'clo5_diff_rate'])
        if df.loc[index, 'clo10_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo10_diff_rate'] = float(df.loc[index, 'clo10_diff_rate'])
        if df.loc[index, 'clo20_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo20_diff_rate'] = float(df.loc[index, 'clo20_diff_rate'])
        if df.loc[index, 'clo40_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo40_diff_rate'] = float(df.loc[index, 'clo40_diff_rate'])

        if df.loc[index, 'clo60_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo60_diff_rate'] = float(df.loc[index, 'clo60_diff_rate'])
        if df.loc[index, 'clo80_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo80_diff_rate'] = float(df.loc[index, 'clo80_diff_rate'])
        if df.loc[index, 'clo100_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo100_diff_rate'] = float(df.loc[index, 'clo100_diff_rate'])
        if df.loc[index, 'clo120_diff_rate'] is not None:
            self.df_all_item.loc[0, 'clo120_diff_rate'] = float(df.loc[index, 'clo120_diff_rate'])

        self.df_all_item.loc[0, 'valuation_profit'] = int(0)

        # 컬럼 중에 nan 값이 있는 경우 0으로 변경 -> 이렇게 안하면 아래 데이터베이스에 넣을 때
        # AttributeError: 'numpy.int64' object has no attribute 'translate' 에러 발생
        self.df_all_item = self.df_all_item.fillna(0)

        self.df_all_item.to_sql('all_item_db', self.engine_simulator, if_exists='append')
        
    # 보유한 종목들을 가져오는 함수
    # sell_date가 0이면 현재 보유 중인 종목이다. 매도를 할 경우 sell_date에 매도 한 날짜가 찍힌다.
    def get_data_from_possessed_item(self):
        sql = "SELECT code_name from all_item_db where sell_date = '%s'"
        return self.engine_simulator.execute(sql % (0)).fetchall()

    # 보유 종복 수 반환 함수
    def get_count_possessed_item(self):
        sql = "SELECT count(*) from all_item_db where sell_date = '%s'"
        return self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

    # 테이블의 존재 여부를 파악하는 함수
    def is_simul_table_exist(self, db_name, table_name):
        sql = "select 1 from information_schema.tables where table_schema = '%s' and table_name = '%s'"
        rows = self.engine_simulator.execute(sql % (db_name, table_name)).fetchall()
        if len(rows) == 1:
            return True
        else:
            return False

    # 일별, 분별 정산 함수
    def check_balance(self):
        # all_item_db가 없으면 check_balance 함수를 나가라
        if self.is_simul_table_exist(self.db_name, "all_item_db") == False:
            return

        # 총 수익 금액 (종목별 평가 금액 합산)
        sql = "SELECT sum(valuation_profit) from all_item_db"
        self.sum_valuation_profit = self.engine_simulator.execute(sql).fetchall()[0][0]
        print("sum_valuation_profit: " + str(self.sum_valuation_profit))

        # 전재산이라고 보면 된다. 현재 총손익 까지 고려했을 때
        self.total_invest_price = self.start_invest_price + self.sum_valuation_profit

        # 현재 총 투자한 금액 계산
        sql = "select sum(item_total_purchase) from all_item_db where sell_date = '%s'"
        self.total_purchase_price = self.engine_simulator.execute(sql % (0)).fetchall()[0][0]
        if self.total_purchase_price is None:
            self.total_purchase_price = 0

        # 매도를 한 종목들 대상 수익 계산
        sql = "select sum(valuation_profit) from all_item_db where sell_date != '%s'"
        self.total_valuation_profit = self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

        if self.total_valuation_profit is None:
            self.total_valuation_profit = 0

        # 현재 투자 가능한 금액(예수금) = (초기자본 + 매도한 종목의 수익) - 현재 총 투자 금액
        self.d2_deposit = self.start_invest_price + self.total_valuation_profit - self.total_purchase_price

    # 시뮬레이팅 할 날짜를 가져 오는 함수
    # 장이 열렸던 날 들을 self.date_rows 에 담기 위해서 gs글로벌의 date값을 대표적으로 가져온 것
    def get_date_for_simul(self):
        sql = "select date from `삼성전자` where date >= '%s' and date <= '%s' group by date"
        self.date_rows = self.engine_daily_craw.execute(sql % (self.simul_start_date, self.simul_end_date)).fetchall()

    # daily_buy_list에 일자 테이블이 존재하는지 확인하는 함수
    def is_date_exist(self, date):
        print("is_date_exist 함수에 들어왔습니다!", date)
        sql = "select 1 from information_schema.tables where table_schema ='daily_buy_list' and table_name = '%s'"
        rows = self.engine_daily_buy_list.execute(sql % (date)).fetchall()
        if len(rows) == 1:
            return True
        else:
            return False

    # 잔액 체크 함수, 잔고가 있으면 True를 반환, 없으면 False를 반환
    def jango_check(self):
        if int(self.d2_deposit) >= (int(self.limit_money) + int(self.invest_unit)):
            return True
        else:
            print("돈부족해서 invest 불가!!!!!!!!")
            return False

    # 출력 함수
    def print_info(self, min_date):
        print("*&*&*&* self.simul_num :" + str(self.simul_num))
        # all_itme_db 테이블이 생성 되어 있으면 보유한 종목 수를 출력
        if self.is_simul_table_exist(self.db_name, "all_item_db"):
            print("simulating 시간: " + str(min_date))
            print("보유종목 수 !!: " + str(self.get_count_possessed_item()))

    # 특정 종목의 시작가를 가져오는 함수(일별)
    def get_now_open_price_by_date(self, code, date):
        sql = "select open from `" + date + "` where code = '%s' group by code"
        open = self.engine_daily_buy_list.execute(sql % (code)).fetchall()
        if len(open) == 1:
            return open[0][0]
        else:
            print("daily_buy_list db의 " + str(date) + " 테이블에 " + str(code) + " 가 존재하지 않는다!")
            return False
        # 테이블의 존재 여부를 파악하는 함수

    # daily_craw 데이터 베이스에서 특정 종목이 존재하는 여부를 파악하는 함수
    def is_daily_craw_table_exist(self, code_name):
        sql = "select 1 from information_schema.tables where table_schema = 'daily_craw' and table_name = '%s'"
        rows = self.engine_daily_craw.execute(sql % (code_name)).fetchall()
        if len(rows) == 1:
            return True
        else:
            print("daily_craw db 에 " + str(code_name) + " 테이블이 존재하지 않는다. !! ")
            return False

    # min_craw 데이터 베이스에서 특정 종목이 존재하는 여부를 파악하는 함수
    def is_min_craw_table_exist(self, code_name):
        sql = "select 1 from information_schema.tables where table_schema = 'min_craw' and table_name = '%s'"
        rows = self.engine_craw.execute(sql % (code_name)).fetchall()
        if len(rows) == 1:
            return True
        else:
            print("min_craw db 에 " + str(code_name) + " 테이블이 존재하지 않는다. !! ")
            return False

    # 분별 현재 누적 거래량 가져오는 함수
    def get_now_volume_by_min(self, code_name, min_date):
        sql = "select sum_volume from `" + code_name + "` where date = '%s' and open != 0 and volume !=0 order by sum_volume desc limit 1"
        rows = self.engine_craw.execute(sql % (min_date)).fetchall()
        if len(rows) == 1:
            return rows[0][0]
        else:
            return False

    # 분별 현재 종가 가져오는 함수
    # (close가 일별 데이터에서는 일별 종가 이지만, 분별 데이터에서의 close는 각 분별에 대한 종가를 의미
    # 즉, 1분 간격으로 변화하는 시세를 가져오는 함수
    def get_now_close_price_by_min(self, code_name, min_date):
        sql = "select close from `" + code_name + "` where date = '{}' and open != 0 and volume !=0 order by sum_volume desc limit 1"
        rows = self.engine_craw.execute(sql.format(min_date)).fetchall()

        if len(rows) == 1:
            return rows[0][0]
        else:
            return False

    # 특정 종목의 종가를 가져오는 함수
    def get_now_close_price_by_date(self, code, date):
        sql = "select close from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False
    # 특정 종목의 clo5 가격을 가져오는 함수
    def get_now_clo5_price_by_date(self, code, date):
        sql = "select clo5 from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False 
    def get_now_clo20_price_by_date(self, code, date):
        sql = "select clo20 from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False               

        # rsi 종가를 가져오는 함수
    def get_now_rsi_price_by_date(self, code_name, date):
        sql = "select close from `" + code_name + "` where DATE_SUB('date', INTERVAL 14 day)  group by code"
        return_price_a += self.engine_daily_craw.execute(sql % (date)).fetchall()

         

 # 특정 종목의 고가를 가져오는 함수
    def get_high_price_by_date(self, code, date):
        sql = "select high from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False


  # 특정 종목의 저가를 가져오는 함수
    def get_low_price_by_date(self, code, date):
        sql = "select low from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False

  # 특정 종목의 시작가를 가져오는 함수
    def get_open_price_by_date(self, code, date):
        sql = "select open from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False                      



    # 특정 종목의 종가를 가져오는 함수 2022-11-28 추가
    def get_now_close_price_by_date_code_name(self, code_name, date):
        sql = "select close from `" + date + "` where code_name = '%s' group by code_name"
        return_price = self.engine_daily_buy_list.execute(sql % (code_name)).fetchall()

        if len(return_price) == 1:
            return return_price[0][0]
        else:
            return False



    # 특정 종목의 어제 종가를 가져오는 함수
    def get_yes_close_price_by_date(self, code, date):
        sql = "select close from `" + date + "` where code = '%s' group by code"
        return_price = self.engine_daily_buy_list.execute(sql % (code)).fetchall()

        if len(return_price) == 1:

            return return_price[0][0]
        else:
            return False

    # 종목의 현재 일자에 대한 주가 정보를 가져 오는 함수
    def get_now_price_by_date(self, code_name, date):
        sql = "select d1_diff_rate, close, open, high, low, volume, clo5, clo10, clo20, clo40, clo60, clo80, clo100, clo120 from `" + date + "` where code_name = '%s' group by code"
        rows = self.engine_daily_buy_list.execute(sql % (code_name)).fetchall()

        if len(rows) == 1:
            return rows
        else:
            return False

    # all_item_db의 보유한 종목에 현재가를 실시간으로 반영하는 함수
    def db_to_all_item_present_price_update_by_min(self, code_name, now_close_price):
        sql = "update all_item_db set present_price = '%s' where code_name = '%s' and sell_date = 0"
        self.engine_simulator.execute(sql % (now_close_price, code_name))

    # 분 마다 보유한 종목의 시세를 업데이트 하는 함수
    def update_all_db_by_min(self, min_date):
        # 매분마다 possess db 가져와야한다
        possessed_code_name = self.get_data_from_possessed_item()
        for j in range(len(possessed_code_name)):
            # 현재 시간의 close 값을 가져온다.
            now_close_price = self.get_now_close_price_by_min(possessed_code_name[j][0], min_date)
            # print("possessed_code_name: ", possessed_code_name[j][0], "now_close_price: ", now_close_price, "min_date", min_date)
            if now_close_price:
                self.db_to_all_item_present_price_update_by_min(possessed_code_name[j][0], now_close_price)
            else:
                # print(min_date + " / " + str(possessed_code_name[j][0]) + " 의 open_price 가 존재하지 않는다")
                continue

    # 보유 중인 종목들의 주가를 일별로 업데이트 하는 함수
    # all_item_db에서 업데이트를 한다.  option = 'ALL' 의미는 인자값을 date 하나만 줬을 때 option에는 기본값으로 ALL을 준다는 의미
    def update_all_db_by_date(self, date, option='ALL'):
        print("update_all_db_by_date 함수에 들어왔다!")
        # 현재 보유 중인 종목 들의 code_name 리스트
        possessed_code_name_list = self.get_data_from_possessed_item()
        if len(possessed_code_name_list) == 0:
            print("현재 보유 중인 종목이 없다 !!!!!")
        for j in range(len(possessed_code_name_list)):
            # 현재 주가를 가져오는 함수
            code_name = possessed_code_name_list[j][0]
            rows = self.get_now_price_by_date(code_name, date)
            if rows == False:
                continue
            d1_diff_rate = rows[0][0]
            close = rows[0][1]
            open = rows[0][2]
            high = rows[0][3]
            low = rows[0][4]
            volume = rows[0][5]
            clo5 = rows[0][6]
            clo10 = rows[0][7]
            clo20 = rows[0][8]
            clo40 = rows[0][9]
            clo60 = rows[0][10]
            clo80 = rows[0][11]
            clo100 = rows[0][12]
            clo120 = rows[0][13]


            # 만약에 open가에 어떤 값이 있으면(True) 현재 주가를 all_item_db에 반영 하기 위해 아래 함수를 들어간다.
            if open:
                self.db_to_all_item_present_price_update(code_name, d1_diff_rate, close, open, high, low, volume, clo5, clo10, clo20,
                                                         clo40, clo60, clo80, clo100, clo120, option)
            else:
                continue

    # 보유 중인 종목들의 주가 이외의 기타 정보들을 업데이트 하는 함수
    def update_all_db_etc(self):
        # valuation_price 업데이트
        sql = f"update all_item_db set valuation_price = round((present_price  * holding_amount) - item_total_purchase * {self.fees_rate} - present_price*holding_amount*{self.fees_rate + self.tax_rate}) where sell_date = '%s'"
        self.engine_simulator.execute(sql % (0))

        # valuation_profit, rate 업데이트
        sql = "update all_item_db set rate= round((valuation_price - item_total_purchase)/item_total_purchase*100,2), valuation_profit =  valuation_price - item_total_purchase where sell_date = '%s';"
        self.engine_simulator.execute(sql % (0))

    # 언제 종목을 팔지(익절, 손절) 결정 하는 알고리즘.
    # !@##############################################################################################################################
    def get_sell_list(self, i):
        print("get_sell_list!!!")
        # 단순히 현재 보유 종목의 수익률이
        # 익절 기준 수익률(self.sell_point) 이 넘거나,
        # 손절 기준 수익률(self.losscut_point) 보다 떨어지면 파는 알고리즘
        if self.sell_list_num == 1:
            # select 할 컬럼은 항상 코드명, 수익률, 매도할 종목의 현재가, 수익(손실)금액
            # sql 첫 번째 라인은 항상 고정
            sql = "SELECT code, rate, present_price,valuation_profit FROM all_item_db WHERE (sell_date = '%s') " \
                  "and (rate>='%s' or rate <= '%s') group by code"
            sell_list = self.engine_simulator.execute(sql % (0, self.sell_point, self.losscut_point)).fetchall()

        # 5 / 20 moving average Death Cross sell list setting 이거나, losscut_point(손절 기준 수익률) 이하로 떨어지면 손절하는 알고리즘
        elif self.sell_list_num == 2:
            sql = "SELECT code, rate, present_price,valuation_profit FROM all_item_db WHERE (sell_date = '%s') " \
                  "and ((clo5 < clo20) or rate <= '%s') group by code"
            sell_list = self.engine_simulator.execute(sql % (0, self.losscut_point)).fetchall()


        # 5 / 40 moving average Death Cross sell list setting 이거나, losscut_point(손절 기준 수익률) 이하로 떨어지면 손절하는 알고리즘
        elif self.sell_list_num == 3:
            sql = "SELECT code, rate, present_price,valuation_profit FROM all_item_db WHERE (sell_date = '%s') " \
                  "and ((clo5 < clo40) or rate <= '%s') group by code"

            sell_list = self.engine_simulator.execute(sql % (0, self.losscut_point)).fetchall()

       #  (clo5 < clo60) or (clo5 < clo40) or (clo5 < clo20)  moving average Death Cross sell list setting 알고리즘
        elif self.sell_list_num == 4:
            sql = "SELECT code,rate,present_price,valuation_profit FROM all_item_db WHERE (sell_date = '%s') " \
                   "and ((clo5 < clo60) or (clo5 < clo40) or (clo5 < clo20)) group by code"
        
            sell_list = self.engine_simulator.execute(sql % (0)).fetchall()

        # Absolute Momentum 전략 (특정일 전 보다 n% 이하로 떨어지면 매도) / code 버전
        elif self.sell_list_num == 5:
            sell_list = []
            sql = "SELECT code, rate, present_price, valuation_profit FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # realtime_daily_buy_list_temp 로 일단 위 조건의 종목을 받는다.
            sell_list_temp = self.engine_simulator.execute(sql).fetchall()
            for row in sell_list_temp:
                code = row[0]
                rate = row[1]
                present_price = row[2]

               # date_before = self.date_rows[i - self.day_before][0]


                date_before_a =self.date_rows[i - 20][0]
                date_before_b =self.date_rows[i - 40][0]
                date_before_c =self.date_rows[i - 60][0]
                date_before_d =self.date_rows[i - 80][0]
                date_before_e =self.date_rows[i - 100][0]
                date_before_f =self.date_rows[i - 120][0]
                date_before_g =self.date_rows[i - 140][0]
                date_before_h =self.date_rows[i - 160][0]
                date_before_i =self.date_rows[i - 180][0]
                date_before_j =self.date_rows[i - 200][0]
                date_before_k =self.date_rows[i - 220][0]
                date_before_l =self.date_rows[i - 240][0]
                # date_before_m =self.date_rows[i - 1 - 200][0]
                # date_before_n =self.date_rows[i - 1 - 200][0]
                
                # 어제 일자 기준 n 일전 종가
                #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                date_before_close_a = self.get_now_close_price_by_date(code, date_before_a)
                date_before_close_b = self.get_now_close_price_by_date(code, date_before_b)
                date_before_close_c = self.get_now_close_price_by_date(code, date_before_c)
                date_before_close_d = self.get_now_close_price_by_date(code, date_before_d)
                date_before_close_e = self.get_now_close_price_by_date(code, date_before_e)
                
                date_before_close_f = self.get_now_close_price_by_date(code, date_before_f)
                date_before_close_g = self.get_now_close_price_by_date(code, date_before_g)
                date_before_close_h = self.get_now_close_price_by_date(code, date_before_h)
                date_before_close_i = self.get_now_close_price_by_date(code, date_before_i)
                date_before_close_j = self.get_now_close_price_by_date(code, date_before_j)

                date_before_close_k = self.get_now_close_price_by_date(code, date_before_k)
                date_before_close_l = self.get_now_close_price_by_date(code, date_before_l)
            
                if date_before_close_a != 0 and date_before_close_a != False and date_before_close_b != 0 and date_before_close_b != False and date_before_close_c != 0 and date_before_close_c != False and date_before_close_d != 0 and date_before_close_d != False and date_before_close_e != 0 and date_before_close_e != False and date_before_close_f != 0 and date_before_close_f != False and date_before_close_g != 0 and date_before_close_g != False and date_before_close_h != 0 and date_before_close_h != False and date_before_close_i != 0 and date_before_close_i != False and date_before_close_j != 0 and date_before_close_j != False and date_before_close_k != 0 and date_before_close_k != False and date_before_close_l != 0 and date_before_close_l != False: 
                    # 모멘텀 계산 : n일전 종가 대비 수익률
                    diff_point_calc = (((present_price - date_before_close_a) / date_before_close_a * 100) +
                                        ((present_price - date_before_close_b) / date_before_close_b * 100) +
                                        ((present_price - date_before_close_c) / date_before_close_c * 100) + 
                                        ((present_price - date_before_close_d) / date_before_close_d * 100) +
                                        ((present_price - date_before_close_e) / date_before_close_e * 100) +
                                        ((present_price - date_before_close_f) / date_before_close_f * 100) +
                                        ((present_price - date_before_close_g) / date_before_close_g * 100) +
                                        ((present_price - date_before_close_h) / date_before_close_h * 100) +
                                        ((present_price - date_before_close_i) / date_before_close_i * 100) +
                                        ((present_price - date_before_close_j) / date_before_close_j * 100) +
                                        ((present_price - date_before_close_k) / date_before_close_k * 100) +
                                        ((present_price - date_before_close_l) / date_before_close_l * 100)) / 12


                    if diff_point_calc < self.sell_diff_point * (-1):                                                           
                         sell_list.append(row)    
                          
                     

        # Absolute Momentum 전략 (특정일 전 보다 n% 이하로 떨어지면 매도) / query version
        elif self.sell_list_num == 6:
            date_before = self.date_rows[i - self.day_before][0]
            sql = "SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit " \
                  "FROM all_item_db ALLDB, daily_buy_list.`" + date_before + "` BEFORE_DAY " \
                    "WHERE ALLDB.code = BEFORE_DAY.code " \
                    "AND ALLDB.sell_date = 0 "\
                    "AND (ALLDB.present_price - BEFORE_DAY.close) / BEFORE_DAY.close * 100 < '%s' "
            sell_list = self.engine_simulator.execute(sql % (self.diff_point * (-1))).fetchall()
        # 2022-10-11 Written by SEONGJAE-YOO (Commits on Oct 11, 2022)
        # Absolute Momentum 전략 + sell_point 추가 + losscut_point 추가 (특정일 전 보다 n% 이하로 떨어지면 매도) / query version
        elif self.sell_list_num == 7:
            date_before = self.date_rows[i - self.day_before][0]
            sql = "SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit " \
                  "FROM all_item_db ALLDB, daily_buy_list.`" + date_before + "` BEFORE_DAY " \
                 "WHERE ALLDB.code = BEFORE_DAY.code " \
                 "AND ALLDB.sell_date = 0 " \
                 "AND ((ALLDB.present_price - BEFORE_DAY.close) / BEFORE_DAY.close * 100 < '%s' " \
                 "OR (ALLDB.rate >= '%s' or ALLDB.rate <= '%s'))"
            sell_list = self.engine_simulator.execute(sql % (self.diff_point * (-1), self.sell_point, self.losscut_point)).fetchall() 

        # Absolute Momentum 전략 + sell_point 추가 + losscut_point 추가 (특정일 전 보다 n% 이하로 떨어지면 매도) / query version
        # group by ALLDB.code DESC : 동일한 코드를 제거한 후  내림차순(DESC)으로 정렬
        # 2022-10-11 Written by SEONGJAE-YOO (Commits on Oct 11, 2022)
        elif self.sell_list_num == 8:
            date_before = self.date_rows[i - self.day_before][0]
            sql = "SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit " \
                  "FROM all_item_db ALLDB, daily_buy_list.`" + date_before + "` BEFORE_DAY " \
                 "WHERE ALLDB.code = BEFORE_DAY.code " \
                 "AND ALLDB.sell_date = '%s' " \
                 "AND ((ALLDB.present_price - BEFORE_DAY.close) / BEFORE_DAY.close * 100 < '%s' " \
                 "OR ((ALLDB.clo5 < ALLDB.clo20) or (ALLDB.clo5 < ALLDB.clo40) or (ALLDB.clo5 < ALLDB.clo60) or ALLDB.rate >= '%s' or ALLDB.rate <= '%s')) " \
                 "ORDER by ALLDB.valuation_profit DESC"       
            sell_list = self.engine_simulator.execute(sql % (0, self.diff_point * (-1), self.sell_point, self.losscut_point)).fetchall()    

        # 볼린저밴드 알고리즘을 위해 추가된 부분 4 (매도 리스트 설정 알고리즘)
        elif self.sell_list_num == 9:   
            sell_list = []
            # 사용하는 이동평균선 기간
            ma_period = 70

            if i > ma_period:
                # all_item_db 에서 매도하지 않고 보유하고 있는 종목 가져오는 쿼리
                sql = """
                        SELECT code, rate, present_price,valuation_profit, code_name
                        FROM all_item_db
                        WHERE (sell_date = '0')
                        group by code
                    """
                sell_list_temp = self.engine_simulator.execute(sql).fetchall()

                # 과매수 포지션 포착
                for item in sell_list_temp:
                    code_name = item.code_name
                    date_rows_yesterday = self.date_rows[i - 1][0]
                    # 보유 중인 종목의 종가 데이터를 가져오는 쿼리
                    bb_sql = f"""
                            SELECT close
                            FROM `{code_name}`
                            WHERE date <= '{date_rows_yesterday}'
                            ORDER BY date DESC limit {ma_period}
                        """
                    df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                    if len(df_close) >= ma_period:
                        # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                        result = BBands(pd.DataFrame(df_close), w=ma_period)
                        # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                        if result:
                            mbb, ubb, lbb, perb, bw = result
                            # perb가 1보다 큰 경우: 종가가 볼린저밴드 상한선 위에 위치 할 경우 매도리스트에 넣는다
                            if perb > 1:
                                sell_list.append(item)
            
        elif self.sell_list_num == 10:

            date_before = self.date_rows[i - self.day_before][0]
            sell_list = []
            ma_period = 20

            if i > ma_period:

                sql = "SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit, ALLDB.code_name " \
                    "FROM all_item_db ALLDB, daily_buy_list.`" + date_before + "` BEFORE_DAY " \
                    "WHERE ALLDB.code = BEFORE_DAY.code " \
                    "AND ALLDB.sell_date = 0 " \
                    "AND ((ALLDB.present_price - BEFORE_DAY.close) / BEFORE_DAY.close * 100 < '%s' " \
                    "OR (ALLDB.rate >= '%s' or ALLDB.rate <= '%s'))"

                sell_list_temp = self.engine_simulator.execute(sql % (self.diff_point * (-1), self.sell_point, self.losscut_point)).fetchall() 

                # 과매수 포지션 포착
                for item in sell_list_temp:
                    code_name = item.code_name
                    date_rows_yesterday = self.date_rows[i - 1][0]
                    # 보유 중인 종목의 종가 데이터를 가져오는 쿼리
                    bb_sql = f"""
                            SELECT close
                            FROM `{code_name}`
                            WHERE date <= '{date_rows_yesterday}'
                            ORDER BY date DESC limit {ma_period}
                        """
                    df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                    if len(df_close) >= ma_period:
                        # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                        result = BBands(pd.DataFrame(df_close), w=ma_period)
                        # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                        if result:
                            mbb, ubb, lbb, perb, bw = result
                            # perb가 1보다 큰 경우: 종가가 볼린저밴드 상한선 위에 위치 할 경우 매도리스트에 넣는다
                            if perb > 1:
                                sell_list.append(item)


        elif self.sell_list_num == 11:

                date_before_a = self.date_rows[i - self.date_before_a][0]
                date_before_b = self.date_rows[i - self.date_before_b][0]

                sell_list = []
                ma_period = 20

                if i > ma_period:

                    sql = f''' 
                         SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit, ALLDB.code_name 
                         FROM all_item_db ALLDB, daily_buy_list.`{date_before_a}` BEFORE_DAY_A, daily_buy_list.`{date_before_b}` BEFORE_DAY_B
                         WHERE ALLDB.code = BEFORE_DAY_A.code
                         AND BEFORE_DAY_A.code = BEFORE_DAY_B.code
                         AND ALLDB.sell_date = 0 
                         AND ((((ALLDB.present_price - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((ALLDB.present_price - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100)) / 2 < {self.diff_point * (-1)}
                         OR (ALLDB.rate >= {self.sell_point} or ALLDB.rate <= {self.losscut_point})) 
                         '''

                    sell_list_temp = self.engine_simulator.execute(sql).fetchall()

                    # 과매수 포지션 포착
                    for item in sell_list_temp:
                        code_name = item.code_name
                        date_rows_yesterday = self.date_rows[i - 1][0]
                        # 보유 중인 종목의 종가 데이터를 가져오는 쿼리
                        bb_sql = f"""
                                SELECT close
                                FROM `{code_name}`
                                WHERE date <= '{date_rows_yesterday}'
                                ORDER BY date DESC limit {ma_period}
                            """
                        df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                        if len(df_close) >= ma_period:
                            # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                            result = BBands(pd.DataFrame(df_close), w=ma_period)
                            # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                            if result:
                                mbb, ubb, lbb, perb, bw = result
                                # perb가 1보다 큰 경우: 종가가 볼린저밴드 상한선 위에 위치 할 경우 매도리스트에 넣는다
                                if perb > 1:
                                    sell_list.append(item)                           

        elif self.sell_list_num == 12:
                    
                    ma_period = 20
    
                    if i < self.date_before_d + 1:
                            sell_list = []
                            pass
                    else:
                            date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                            date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                            date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                            date_before_d =self.date_rows[i - 1 - self.date_before_d][0]    
                    
                    # sql = f''' 
                    #         SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit, ALLDB.code_name
                    #         FROM all_item_db ALLDB, daily_buy_list.`{date_before_a}` BEFORE_DAY_A, daily_buy_list.`{date_before_b}` BEFORE_DAY_B ,daily_buy_list.`{date_before_c}` BEFORE_DAY_C, daily_buy_list.`{date_before_d}` BEFORE_DAY_D
                    #         WHERE ALLDB.code = BEFORE_DAY_A.code = BEFORE_DAY_B.code = BEFORE_DAY_C.code = BEFORE_DAY_D.code  
                    #         AND ALLDB.sell_date = 0
                    #         OR (((ALLDB.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((ALLDB.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((ALLDB.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((ALLDB.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 < {self.diff_point * (-1)}                                  
                    #         '''
                    if i > ma_period:
                        

                        sql =  "SELECT ALLDB.* " \
                        "FROM all_item_db ALLDB, daily_buy_list.`" + date_before_a + "` BEFORE_DAY_A, daily_buy_list.`" + date_before_b + "` BEFORE_DAY_B, daily_buy_list.`" + date_before_c + "` BEFORE_DAY_C , daily_buy_list.`" + date_before_d + "` BEFORE_DAY_D " \
                        "WHERE ALLDB.code = BEFORE_DAY_A.code = BEFORE_DAY_B.code = BEFORE_DAY_C.code =BEFORE_DAY_D.code  " \
                        "AND ALLDB.sell_date = 0 " \
                        "AND ((ALLDB.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100 + (ALLDB.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100 + (ALLDB.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100 + (ALLDB.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100) / 4 < '%s' " 
                        
                        sell_list = self.engine_simulator.execute(sql % (self.diff_point * (-1))).fetchall() 
                        # sql = f''' 
                        #     SELECT ALLDB.code, ALLDB.rate, ALLDB.present_price, ALLDB.valuation_profit, ALLDB.code_name 
                        #     FROM all_item_db ALLDB, daily_buy_list.`{date_before_a}` BEFORE_DAY_A, daily_buy_list.`{date_before_b}` BEFORE_DAY_B ,daily_buy_list.`{date_before_c}` BEFORE_DAY_C, daily_buy_list.`{date_before_d}` BEFORE_DAY_D
                        #     WHERE ALLDB.code = BEFORE_DAY_A.code = BEFORE_DAY_B.code = BEFORE_DAY_C.code = BEFORE_DAY_D.code  
                        #     AND ALLDB.sell_date = 0
                        #     OR (ALLDB.rate >= {self.sell_point} or ALLDB.rate <= {self.losscut_point})
                        #     '''
                        #    # OR (((ALLDB.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100) + ((ALLDB.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100) + ((ALLDB.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100) + ((ALLDB.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100)) / 4 < {self.diff_point * (-1)}                                  
                           
                        #     # OR (ALLDB.clo5 < ALLDB.clo20) or (ALLDB.clo5 < ALLDB.clo40) or (ALLDB.clo5 < ALLDB.clo60))   
                        #     # OR (ALLDB.rate >= {self.sell_point} or ALLDB.rate <= {self.losscut_point}))    
                        #     # (ALLDB.rate >= {self.sell_point} or ALLDB.rate <= {self.losscut_point})) 
                        # sell_list = self.engine_simulator.execute(sql).fetchall()
                        
                        # 과매수 포지션 포착
                        for item in sell_list:
                            code_name = item.code_name
                            date_rows_yesterday = self.date_rows[i - 1][0]
                            # 보유 중인 종목의 종가 데이터를 가져오는 쿼리
                            bb_sql = f"""
                                    SELECT close
                                    FROM `{code_name}`
                                    WHERE date <= '{date_rows_yesterday}'
                                    ORDER BY date DESC limit {ma_period}
                                """
                            df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                            if len(df_close) >= ma_period:
                                # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                                result = BBands(pd.DataFrame(df_close), w=ma_period)
                                # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                                if result:
                                    mbb, ubb, lbb, perb, bw = result
                                    # perb가 1과 같거나 작은 경우: 종가가 볼린저밴드 상한선 아래에 위치 할 경우 매도리스트에서 제외한다.
                                    if perb <= 1:
                                        sell_list.remove(item) 
                                    
                                    # # perb가 1보다 큰 경우: 종가가 볼린저밴드 상한선 위에 위치 할 경우 매도리스트에 넣는다
                                    # if perb > 1:
                                    #     sell_list.append(item)
        elif self.sell_list_num == 13:
            
            if i < self.date_before_d + 1:
                sell_list = []
                pass
            else:

                date_before_a =self.date_rows[i - 1 - self.date_before_a][0]
                date_before_b =self.date_rows[i - 1 - self.date_before_b][0]
                date_before_c =self.date_rows[i - 1 - self.date_before_c][0]
                date_before_d =self.date_rows[i - 1 - self.date_before_d][0]

                sql = "SELECT ALLDB.* " \
                        "FROM all_item_db ALLDB, daily_buy_list.`" + date_before_a + "` BEFORE_DAY_A, daily_buy_list.`" + date_before_b + "` BEFORE_DAY_B, daily_buy_list.`" + date_before_c + "` BEFORE_DAY_C , daily_buy_list.`" + date_before_d + "` BEFORE_DAY_D " \
                        "WHERE ALLDB.code = BEFORE_DAY_A.code = BEFORE_DAY_B.code = BEFORE_DAY_C.code =BEFORE_DAY_D.code  " \
                        "AND ALLDB.sell_date = 0 " \
                        "AND ((ALLDB.close - BEFORE_DAY_A.close) / BEFORE_DAY_A.close * 100 + (ALLDB.close - BEFORE_DAY_B.close) / BEFORE_DAY_B.close * 100 + (ALLDB.close - BEFORE_DAY_C.close) / BEFORE_DAY_C.close * 100 + (ALLDB.close - BEFORE_DAY_D.close) / BEFORE_DAY_D.close * 100) / 4 < '%s' " 
                        
                sell_list = self.engine_simulator.execute(sql % (self.diff_point * (-0.8))).fetchall() 
        elif self.sell_list_num == 14:
            sell_list = []
            # 사용하는 이동평균선 기간
            ma_period = 70

            if i > ma_period:
                # all_item_db 에서 매도하지 않고 보유하고 있는 종목 가져오는 쿼리
                # sql = "SELECT code, rate, present_price,valuation_profit, code_name FROM all_item_db WHERE (sell_date = '%s') " \
                #       "and ((clo5 < clo20) or rate <= '%s') group by code"
                # sell_list_temp = self.engine_simulator.execute(sql % (0, self.losscut_point)).fetchall()
                sql = "SELECT code, rate, present_price,valuation_profit, code_name FROM all_item_db WHERE (sell_date = '%s') " \
                  "and (rate>='%s' or rate <= '%s') group by code"
                sell_list_temp = self.engine_simulator.execute(sql % (0, self.sell_point, self.losscut_point)).fetchall()

                # 과매수 포지션 포착
                for item in sell_list_temp:
                    code_name = item.code_name
                    date_rows_yesterday = self.date_rows[i - 1][0]
                    # 보유 중인 종목의 종가 데이터를 가져오는 쿼리
                    bb_sql = f"""
                            SELECT close
                            FROM `{code_name}`
                            WHERE date <= '{date_rows_yesterday}'
                            ORDER BY date DESC limit {ma_period}
                        """
                    df_close = self.engine_daily_craw.execute(bb_sql).fetchall()

                    if len(df_close) >= ma_period:
                        # 데이터프레임으로 종가리스트를 담아서 trading_algorithms.py 파일에 존재하는 BBands 함수에 보내주는 코드
                        result = BBands(pd.DataFrame(df_close), w=ma_period)
                        # result가 false가 아닐 경우 볼린저밴드의 수치를 가지고 알고리즘을 구현
                        if result:
                            mbb, ubb, lbb, perb, bw = result
                            # perb가 1보다 큰 경우: 종가가 볼린저밴드 상한선 위에 위치 할 경우 매도리스트에 넣는다
                            if perb > 0.5:
                                sell_list.append(item)


        # sell ai 기능 적용
        elif self.sell_list_num == 15:
            #sell_list_1 = ''
            sell_list = []

            sell_ai_settings = {
                "model": None,   
                "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                "test_size": 0.3,
                "batch_size": 32,
                "epochs": 100,
                "ratio_cut": -2,
                "table": "daily_craw",
                "is_used_predicted_close" : True #false는 단한종목도 사지 않는다.
            }

            tr_engine = create_training_engine(sell_ai_settings['table']) 
           
            sql_temp = "SELECT code, rate, close, valuation_profit,code_name FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # realtime_daily_buy_list_temp 로 일단 위 조건의 종목을 받는다.
            sell_list_temp = self.engine_simulator.execute(sql_temp).fetchall()
            for row in sell_list_temp:
                code = row[0]
                
                code_name= row.code_name
                # code_name = row[1]
                # code_name = pd.DataFrame([code_name],columns=['code_name'])
                rate = row[1]
                close = row[2]

               # date_before = self.date_rows[i - self.day_before][0]


                date_before_a =self.date_rows[i - 1 - 20][0]
                date_before_b =self.date_rows[i - 1 - 40][0]
                date_before_c =self.date_rows[i - 1 - 60][0]
                date_before_d =self.date_rows[i - 1 - 80][0]
                date_before_e =self.date_rows[i - 1 - 100][0]
                date_before_f =self.date_rows[i - 1 - 120][0]
                date_before_g =self.date_rows[i - 1 - 140][0]
                date_before_h =self.date_rows[i - 1 - 160][0]
                date_before_i =self.date_rows[i - 1 - 180][0]
                date_before_j =self.date_rows[i - 1 - 200][0]
                date_before_k =self.date_rows[i - 1 - 220][0]
                date_before_l =self.date_rows[i - 1 - 240][0]
                # date_before_m =self.date_rows[i - 1 - 200][0]
                # date_before_n =self.date_rows[i - 1 - 200][0]
                
                # 어제 일자 기준 n 일전 종가
                #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                date_before_close_a = self.get_now_clo5_price_by_date(code, date_before_a)
                # date_before_close_b = self.get_now_close_price_by_date(code, date_before_b)
                # date_before_close_c = self.get_now_close_price_by_date(code, date_before_c)
                # date_before_close_d = self.get_now_close_price_by_date(code, date_before_d)
                # date_before_close_e = self.get_now_close_price_by_date(code, date_before_e)
                
                # date_before_close_f = self.get_now_close_price_by_date(code, date_before_f)
                # date_before_close_g = self.get_now_close_price_by_date(code, date_before_g)
                # date_before_close_h = self.get_now_close_price_by_date(code, date_before_h)
                # date_before_close_i = self.get_now_close_price_by_date(code, date_before_i)
                # date_before_close_j = self.get_now_close_price_by_date(code, date_before_j)

                # date_before_close_k = self.get_now_close_price_by_date(code, date_before_k)
                # date_before_close_l = self.get_now_close_price_by_date(code, date_before_l)
            
                if date_before_close_a != 0 and date_before_close_a != False and date_before_close_b != 0 and date_before_close_b != False and date_before_close_c != 0 and date_before_close_c != False and date_before_close_d != 0 and date_before_close_d != False and date_before_close_e != 0 and date_before_close_e != False and date_before_close_f != 0 and date_before_close_f != False and date_before_close_g != 0 and date_before_close_g != False and date_before_close_h != 0 and date_before_close_h != False and date_before_close_i != 0 and date_before_close_i != False and date_before_close_j != 0 and date_before_close_j != False and date_before_close_k != 0 and date_before_close_k != False and date_before_close_l != 0 and date_before_close_l != False: 
                    # 모멘텀 계산 : n일전 종가 대비 수익률
                    diff_point_calc = ((close - date_before_close_a) / date_before_close_a * 100)


                    if diff_point_calc < self.sell_diff_point * (-1):                                                           
                            #sell_list_a = self.get_now_close_price_by_date(code, )    
                                #code_name= row.code_name    
                            #feature_columns = ["close", "volume", "open", "high", "low"]
                        #filtered_list = []
                        #for code_name, in row:
                            
                                feature_columns = ["close", "volume", "open", "high", "low"]
                                #print(f"{code_name} 종목 분석 중....")
                                # sql = """
                                #     SELECT {} FROM `{}`
                                #     WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                                # """.format(','.join(feature_columns), code_name, until=datetime.datetime.today())
                                # # pandas(pd) read_sql 을 사용하면 sql, engine을 넘겼을 때 return 값을 바로 데이터프레임으로 받을 수 있음
                                # sell_df = pd.read_sql(sql, tr_engine)
                                date_rows_yesterday = self.date_rows[i-1][0]
                                # sell_sql = f"""
                                #         SELECT close, volume, open, high, low
                                #         FROM `{code_name}`
                                #         WHERE date <= '{date_rows_yesterday}'
                                        
                                #         """
                                # sell_df_1 = self.engine_daily_craw.execute(sell_sql).fetchall()

                                sql = """
                                        SELECT {} FROM `{}`
                                        WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                                """.format(','.join(feature_columns), code_name, date_rows_yesterday)
                                sell_df = pd.read_sql(sql, tr_engine)
                                
                                # 데이터가 2000개(2000일 or 2000분)가 넘지 않으면 예측도가 떨어지기 때문에 필터링
                                if len(sell_df) < 1:
                                    # filtered_list.append(code_name)
                                    print(f"테스트 데이터가 적어요")
                                    continue
                                try:
                                    if 1<= len(sell_df) <=5000:
                                        sell_ai_settings['model'] = CNN_Attention_BiLSTM_Version9()
                                        filtered = sell_list_ai(sell_df, sell_ai_settings)
                                    elif len(sell_df) >5000:                          
                                        sell_ai_settings['model'] = CNN_Attention_BiLSTM_Version9()
                                        filtered = sell_list_ai_v2(sell_df, sell_ai_settings)
                                    #filtered = sell_list_ai(sell_df, sell_ai_settings)
                                except (DataNotEnough, ValueError):
                                    print(f"테스트 데이터가 적어요")
                                    #filtered_list.append(code_name)
                                    continue

                                print(code_name)

                                # filtered가 True 이면 sell_list에 해당 종목을 append
                                if filtered:
                                    print(f"기준에 부합하지 않으므로 제외")
                                    sell_list.append(row)
                                    # sell_list_1 = []
                                    #sell_list_1.append(code_name)
                                 
                                    # sql = "SELECT code, rate, present_price, valuation_profit FROM all_item_db WHERE sell_date = 0 and code_name = '%s' group by code"  

                                    # # realtime_daily_buy_list_temp 로 일단 위 조건의 종목을 받는다.
                                    # sell_list += self.engine_simulator.execute(sql%(code_name)).fetchall()
                        
                                # # filtered_list에 있는 종목들을 realtime_daily_buy_list(매수리스트)에서 제거
                                # # 모든 조건문에서 filtered_list를 생성해줘야 함
                                # if len(sell_df) > 0:
                                #     sell_list.append(sell_df) 
        # sell AI 기능 적용
        elif self.sell_list_num == 16:
            #sell_list_1 = ''
            sell_list = []

            sell_ai_settings = {
                "model": None,   
                "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                "test_size": 0.3,
                "batch_size": 32,
                "epochs": 100,
                "ratio_cut": -1,
                "table": "daily_craw",
                "is_used_predicted_close" : True #false는 단한종목도 사지 않는다.
            }

            tr_engine = create_training_engine(sell_ai_settings['table']) 
           
            sql_temp = "SELECT code, rate, close, valuation_profit,code_name FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # 
            sell_list_temp = self.engine_simulator.execute(sql_temp).fetchall()
            for row in sell_list_temp:
                code = row[0]
                
                code_name= row.code_name
                # code_name = row[1]
                # code_name = pd.DataFrame([code_name],columns=['code_name'])
                rate = row[1]
                close = row[2]

               # date_before = self.date_rows[i - self.day_before][0]
            
            

                                                                               
                            
                feature_columns = ["close", "volume", "open", "high", "low"]
                               
                date_rows_yesterday = self.date_rows[i-1][0]
                # sell_sql = f"""
                #         SELECT close, volume, open, high, low
                #         FROM `{code_name}`
                #         WHERE date <= '{date_rows_yesterday}'
                        
                #         """
                # sell_df_1 = self.engine_daily_craw.execute(sell_sql).fetchall()

                sql = """
                        SELECT {} FROM `{}`
                        WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, date_rows_yesterday)
                sell_df = pd.read_sql(sql, tr_engine)
                
                # 데이터가 1개(1일 or 1분)가 넘지 않으면 예측도가 떨어지기 때문에 필터링
                if len(sell_df) < 1:
                    # filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어요")
                    continue
                try:
                    if 1<= len(sell_df) <=5000:
                        sell_ai_settings['model'] = CNN_Attention_BiLSTM_Version27()
                        filtered = sell_list_ai(sell_df, sell_ai_settings)
                    elif len(sell_df) >5000:                          
                        sell_ai_settings['model'] = CNN_Attention_BiLSTM_Version27()
                        filtered = sell_list_ai_v2(sell_df, sell_ai_settings)
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어요")
                    #filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 sell_list에 해당 종목을 append , # -2>= -3 
                if filtered:
                    print(f"기준에 부합되므로 추가됨")
                    sell_list.append(row)
                                            
        elif self.sell_list_num == 17:
            sell_list = []
            sql = "SELECT code, rate, present_price, valuation_profit FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # realtime_daily_buy_list_temp 로 일단 위 조건의 종목을 받는다.
            sell_list_temp = self.engine_simulator.execute(sql).fetchall()
            for row in sell_list_temp:
                code = row[0]
                rate = row[1]
                present_price = row[2]

               # date_before = self.date_rows[i - self.day_before][0]


                date_before_a =self.date_rows[i - 20][0]
                date_before_b =self.date_rows[i - 40][0]
                date_before_c =self.date_rows[i - 60][0]
                date_before_d =self.date_rows[i - 80][0]
                date_before_e =self.date_rows[i - 100][0]
                date_before_f =self.date_rows[i - 120][0]
                date_before_g =self.date_rows[i - 140][0]
                date_before_h =self.date_rows[i - 160][0]
                date_before_i =self.date_rows[i - 180][0]
                date_before_j =self.date_rows[i - 200][0]
                date_before_k =self.date_rows[i - 220][0]
                date_before_l =self.date_rows[i - 240][0]
                # date_before_m =self.date_rows[i - 1 - 200][0]
                # date_before_n =self.date_rows[i - 1 - 200][0]
                
                # 어제 일자 기준 n 일전 종가
                #date_before_close = self.get_now_close_price_by_date_code_name(code_name, date_before)

                date_before_close_a = self.get_now_clo5_price_by_date(code, date_before_a)
                # date_before_close_b = self.get_now_close_price_by_date(code, date_before_b)
                # date_before_close_c = self.get_now_close_price_by_date(code, date_before_c)
                # date_before_close_d = self.get_now_close_price_by_date(code, date_before_d)
                # date_before_close_e = self.get_now_close_price_by_date(code, date_before_e)
                
                # date_before_close_f = self.get_now_close_price_by_date(code, date_before_f)
                # date_before_close_g = self.get_now_close_price_by_date(code, date_before_g)
                # date_before_close_h = self.get_now_close_price_by_date(code, date_before_h)
                # date_before_close_i = self.get_now_close_price_by_date(code, date_before_i)
                # date_before_close_j = self.get_now_close_price_by_date(code, date_before_j)

                # date_before_close_k = self.get_now_close_price_by_date(code, date_before_k)
                # date_before_close_l = self.get_now_close_price_by_date(code, date_before_l)
            
                if date_before_close_a != 0 and date_before_close_a != False: 
                    # 모멘텀 계산 : n일전 종가 대비 수익률
                    diff_point_calc = ((present_price - date_before_close_a) / date_before_close_a * 100)


                    if diff_point_calc < self.sell_diff_point * (-1):                                                           
                         sell_list.append(row)

        elif self.sell_list_num == 18:
            #sell_list_1 = ''
            sell_list = []

            sell_ai_settings = {
                "model": None,   
                "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                "test_size": 0.3,
                "batch_size": 32,
                "epochs": 100,
                "ratio_cut": -1,
                "table": "daily_craw",
                "is_used_predicted_close" : True
            }

            tr_engine = create_training_engine(sell_ai_settings['table']) 
           
            sql_temp = "SELECT code, rate, present_price, valuation_profit,code_name FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # 
            sell_list_temp = self.engine_simulator.execute(sql_temp).fetchall()
            for row in sell_list_temp:
                code = row[0]
                
                code_name= row.code_name
                # code_name = row[1]
                # code_name = pd.DataFrame([code_name],columns=['code_name'])
                #rate = row[1]
                #close = row[2]

               # date_before = self.date_rows[i - self.day_before][0]
            
            

                                                                               
                            
                feature_columns = ["close", "volume", "open", "high", "low"]
                               
                date_rows_yesterday = self.date_rows[i-1][0]
                # sell_sql = f"""
                #         SELECT close, volume, open, high, low
                #         FROM `{code_name}`
                #         WHERE date <= '{date_rows_yesterday}'
                        
                #         """
                # sell_df_1 = self.engine_daily_craw.execute(sell_sql).fetchall()

                sql = """
                        SELECT {} FROM `{}`
                        WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, date_rows_yesterday)
                sell_df = pd.read_sql(sql, tr_engine)
                
                # 데이터가 없으면 필터링
                if len(sell_df) < 1:
                    # filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어요")
                    continue
                try:
                    if 1<= len(sell_df) <=5000:
                        sell_ai_settings['model'] = CNN_Attention_BiLSTM_Version27()
                        filtered = sell_list_ai(sell_df, sell_ai_settings)
                    elif len(sell_df) >5000:                          
                        sell_ai_settings['model'] = CNN_Attention_BiLSTM_Version27()
                        filtered = sell_list_ai_v2(sell_df, sell_ai_settings)
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어요")
                    #filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 sell_list에 해당 종목을 append 
                if filtered:
                    print(f"기준에 부합되므로 추가됨")
                    sell_list.append(row)

        elif self.sell_list_num == 19:
            #sell_list_1 = ''
            sell_list = []

            sell_ai_settings = {
                "model": None,   
                "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                "test_size": 0.3,
                "batch_size": 32,
                "epochs": 100,
                "ratio_cut": -1,
                "table": "daily_craw",
                "is_used_predicted_close" : True
            }

            tr_engine = create_training_engine(sell_ai_settings['table']) 
           
            sql_temp = "SELECT code, rate, present_price, valuation_profit,code_name FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # 
            sell_list_temp = self.engine_simulator.execute(sql_temp).fetchall()
            for row in sell_list_temp:
                code = row[0]
                
                code_name= row.code_name
                # code_name = row[1]
                # code_name = pd.DataFrame([code_name],columns=['code_name'])
                #rate = row[1]
                #close = row[2]

               # date_before = self.date_rows[i - self.day_before][0]
            
            

                                                                               
                            
                feature_columns = ["close", "volume", "open", "high", "low"]
                               
                date_rows_yesterday = self.date_rows[i-1][0]
                # sell_sql = f"""
                #         SELECT close, volume, open, high, low
                #         FROM `{code_name}`
                #         WHERE date <= '{date_rows_yesterday}'
                        
                #         """
                # sell_df_1 = self.engine_daily_craw.execute(sell_sql).fetchall()

                sql = """
                        SELECT {} FROM `{}`
                        WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, date_rows_yesterday)
                sell_df = pd.read_sql(sql, tr_engine)
                
                # 데이터가 없으면 필터링
                if len(sell_df) < 1:
                    # filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어요")
                    continue
                try:
                    if 1<= len(sell_df) <=5000:
                        sell_ai_settings['model'] = BiGRU_CNN_BiLSTM_Attention_version2()
                        filtered = sell_list_ai(sell_df, sell_ai_settings)
                    elif len(sell_df) >5000:                          
                        sell_ai_settings['model'] = BiGRU_CNN_BiLSTM_Attention_version2()
                        filtered = sell_list_ai_v2(sell_df, sell_ai_settings)
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어요")
                    #filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 sell_list에 해당 종목을 append 
                if filtered:
                    print(f"기준에 부합되므로 추가됨")
                    sell_list.append(row)

        elif self.sell_list_num == 20:
            #sell_list_1 = ''
            sell_list = []

            sell_ai_settings = {
                "model": None,   
                "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                "test_size": 0.3,
                "batch_size": 32,
                "epochs": 100,
                "ratio_cut": -1,
                "table": "daily_craw",
                "is_used_predicted_close" : True
            }

            tr_engine = create_training_engine(sell_ai_settings['table']) 
           
            sql_temp = "SELECT code, rate, present_price, valuation_profit,code_name FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # 
            sell_list_temp = self.engine_simulator.execute(sql_temp).fetchall()
            for row in sell_list_temp:
                code = row[0]
                
                code_name= row.code_name
                # code_name = row[1]
                # code_name = pd.DataFrame([code_name],columns=['code_name'])
                #rate = row[1]
                #close = row[2]

               # date_before = self.date_rows[i - self.day_before][0]
            
            

                                                                               
                            
                feature_columns = ["close", "volume", "open", "high", "low"]
                               
                date_rows_yesterday = self.date_rows[i-1][0]
                # sell_sql = f"""
                #         SELECT close, volume, open, high, low
                #         FROM `{code_name}`
                #         WHERE date <= '{date_rows_yesterday}'
                        
                #         """
                # sell_df_1 = self.engine_daily_craw.execute(sell_sql).fetchall()

                sql = """
                        SELECT {} FROM `{}`
                        WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, date_rows_yesterday)
                sell_df = pd.read_sql(sql, tr_engine)
                
                # 데이터가 없으면 필터링
                if len(sell_df) < 1:
                    # filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어요")
                    continue
                try:
                    if 1<= len(sell_df) <=5000:
                        sell_ai_settings['model'] = BiLSTM_Attention_CNN_version2()
                        filtered = sell_list_ai(sell_df, sell_ai_settings)
                    elif len(sell_df) >5000:                          
                        sell_ai_settings['model'] = BiLSTM_Attention_CNN_version2()
                        filtered = sell_list_ai_v2(sell_df, sell_ai_settings)
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어요")
                    #filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 sell_list에 해당 종목을 append 
                if filtered:
                    print(f"기준에 부합되므로 추가됨")
                    sell_list.append(row)

        elif self.sell_list_num == 21:
            #sell_list_1 = ''
            sell_list = []

            sell_ai_settings = {
                "model": None,   
                "n_steps": 1, # 시퀀스 데이터를 몇개씩 담을지 설정       
                "lookup_step": 1, #단위 :(일/분) 몇 일(분) 뒤의 종가를 예측 할 것 인지 설정 : daily_craw -> 일 / min_craw -> 분
                "test_size": 0.3,
                "batch_size": 32,
                "epochs": 100,
                "ratio_cut": -1,
                "table": "daily_craw",
                "is_used_predicted_close" : True
            }

            tr_engine = create_training_engine(sell_ai_settings['table']) 
           
            sql_temp = "SELECT code, rate, present_price, valuation_profit,code_name FROM all_item_db WHERE sell_date = 0 group by code"  
                  
            # 
            sell_list_temp = self.engine_simulator.execute(sql_temp).fetchall()
            for row in sell_list_temp:
                code = row[0]
                
                code_name= row.code_name
                # code_name = row[1]
                # code_name = pd.DataFrame([code_name],columns=['code_name'])
                #rate = row[1]
                #close = row[2]

               # date_before = self.date_rows[i - self.day_before][0]
            
            

                                                                               
                            
                feature_columns = ["close", "volume", "open", "high", "low"]
                               
                date_rows_yesterday = self.date_rows[i-1][0]
                # sell_sql = f"""
                #         SELECT close, volume, open, high, low
                #         FROM `{code_name}`
                #         WHERE date <= '{date_rows_yesterday}'
                        
                #         """
                # sell_df_1 = self.engine_daily_craw.execute(sell_sql).fetchall()

                sql = """
                        SELECT {} FROM `{}`
                        WHERE STR_TO_DATE(date, '%Y%m%d%H%i') <= '{}'
                """.format(','.join(feature_columns), code_name, date_rows_yesterday)
                sell_df = pd.read_sql(sql, tr_engine)
                
                # 데이터가 없으면 필터링
                if len(sell_df) < 1:
                    # filtered_list.append(code_name)
                    print(f"테스트 데이터가 적어요")
                    continue
                try:
                    if 1<= len(sell_df) <=5000:
                        sell_ai_settings['model'] = CNN_BiLSTM_Attention_version2()
                        filtered = sell_list_ai(sell_df, sell_ai_settings)
                    elif len(sell_df) >5000:                          
                        sell_ai_settings['model'] = CNN_BiLSTM_Attention_version2()
                        filtered = sell_list_ai_v2(sell_df, sell_ai_settings)
                except (DataNotEnough, ValueError):
                    print(f"테스트 데이터가 적어요")
                    #filtered_list.append(code_name)
                    continue

                print(code_name)

                # filtered가 True 이면 sell_list에 해당 종목을 append 
                if filtered:
                    print(f"기준에 부합되므로 추가됨")
                    sell_list.append(row)

        ##################################################################################################################################################################################################################
        else:
            print(f"{self.simul_num}번 알고리즘에 대한 self.sell_list_num 설정이 비었습니다. variable_setting 함수에서 self.sell_list_num을 확인해주세요.")
            sys.exit(1)

        return sell_list

    # 실제로 매도를 하는 함수 (매도 한 결과를 all_item_db에 반영)
    def sell_send_order(self, min_date, sell_price, sell_rate, code):
        # print("sell send order")
        sql = "UPDATE all_item_db SET sell_date= '%s', sell_price ='%s' ,sell_rate ='%s' WHERE code='%s' and sell_date = '%s' " \
              "ORDER BY buy_date desc LIMIT 1"
        self.engine_simulator.execute(sql % (min_date, sell_price, sell_rate, code, 0))
        # 매도 후 정산
        self.check_balance()

    # 매도를 하기 위한 함수
    def auto_trade_sell_stock(self, date, _i):
        # 매도 할 리스트를 가져오는 함수
        sell_list = self.get_sell_list(_i)
        for i in range(len(sell_list)):
            # 코드명
            get_sell_code = sell_list[i][0]
            # 수익률
            get_sell_rate = sell_list[i][1]
            # 종목의 현재 주가
            get_present_price = sell_list[i][2]
            # 수익(손실) 금액 (종목의 순수익, 순손실 금액)
            valuation_profit = sell_list[i][3]

            if get_sell_rate < 0:
                print("손절 매도!!!!$$$$$$$$$$$ 수익: " + str(valuation_profit) + " / 수익률 : " + str(
                    get_sell_rate) + " / 종목코드: " + str(get_sell_code) + " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

            else:
                print("익절 매도!!!!$$$$$$$$$$$ 수익: " + str(valuation_profit) + " / 수익률 : " + str(
                    get_sell_rate) + " / 종목코드: " + str(get_sell_code) + " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

            # 실제로 매도를 하는 함수 (매도 한 결과를 all_item_db에 반영)
            self.sell_send_order(date, get_present_price, get_sell_rate, get_sell_code)

    # 몇개의 주를 살지 계산해주는 함수
    def buy_num_count(self, invest_unit, present_price):
        # jackbot("******************* buy_num_count!!!")
        return int(int(invest_unit) / int(present_price))

    # 금일 수익 계산 함수
    def get_today_profit(self, date):
        # jackbot("******************* get_today_profit!!!")
        sql = "SELECT sum(valuation_profit) from all_item_db where sell_date like '%s'"
        return self.engine_simulator.execute(sql % ("%%" + date + "%%")).fetchall()[0][0]

    # 총 매입금액 계산 함수
    def get_sum_item_total_purchase(self):

        # jackbot("******************* get_sum_item_total_purchase!!!")
        sql = "SELECT sum(item_total_purchase) from all_item_db where sell_date = '%s'"
        rows = self.engine_simulator.execute(sql % (0)).fetchall()[0][0]
        if rows is not None:
            return rows
        else:
            return 0

    # 총평가금액 계산 함수
    def get_sum_valuation_price(self):
        sql = "SELECT sum(valuation_price) from all_item_db where sell_date = '%s'"
        rows = self.engine_simulator.execute(sql % (0)).fetchall()[0][0]
        if rows is not None:
            return rows
        else:
            return 0

    # 오늘 일자 익절 종목 수
    def get_today_profitcut_count(self, date):
        sql = "SELECT count(code) from all_item_db where sell_date like '%s' and sell_rate>='%s'"
        return self.engine_simulator.execute(sql % ("%%" + date + "%%", 0)).fetchall()[0][0]

    # 오늘 일자 손절 종목 수
    def get_today_losscut_count(self, date):
        sql = "SELECT count(code) from all_item_db where sell_date like '%s' and sell_rate<'%s'"
        return self.engine_simulator.execute(sql % ("%%" + date + "%%", 0)).fetchall()[0][0]

    # 오늘 일자 매도금액
    def get_sum_today_sell_price(self, date):
        sql = "SELECT sum(valuation_price) from all_item_db where sell_date like '%s'"
        return self.engine_simulator.execute(sql % ("%%" + date + "%%")).fetchall()[0][0]

    # 오늘 일자 익절 종목 대상 수익
    def get_sum_today_profitcut(self, date):
        sql = "SELECT sum(valuation_profit) from all_item_db where sell_date like '%s' and valuation_profit >= '%s' "
        return self.engine_simulator.execute(sql % ("%%" + date + "%%", 0)).fetchall()[0][0]

    # 오늘 일자 손절 종목 대상 손실 금액
    def get_sum_today_losscut(self, date):
        sql = "SELECT sum(valuation_profit) from all_item_db where sell_date like '%s' and valuation_profit < '%s' "
        return self.engine_simulator.execute(sql % ("%%" + date + "%%", 0)).fetchall()[0][0]

    # 총 익절 종목 대상 수익
    def get_sum_total_profitcut(self):
        sql = "SELECT sum(valuation_profit) from all_item_db where sell_date != 0 and valuation_profit >= '%s' "
        return self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

    # 총 손절 종목 대상 손실 금액
    def get_sum_total_losscut(self):
        sql = "SELECT sum(valuation_profit) from all_item_db where sell_date != 0 and valuation_profit < '%s' "
        return self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

    # 전체 일자 익절한 종목 수
    def get_sum_total_profitcut_count(self):
        # jackbot("******************* get_sum_total_profitcut_count!!!")
        sql = "select count(code) from all_item_db where sell_date != 0 and valuation_profit >= '%s'"
        return self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

    # 전체 일자 손절한 종목 수
    def get_sum_total_losscut_count(self):
        # jackbot("******************* get_sum_total_losscut_count!!!")
        sql = "select count(code) from all_item_db where sell_date != 0 and valuation_profit < '%s' "
        return self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

    # jango_data의 저장 된 일자 반환 함수
    def get_len_jango_data_date(self):

        sql = "select date from jango_data"
        rows = self.engine_simulator.execute(sql).fetchall()

        return len(rows)

    # 총 보유한 종목 수
    def get_total_possess_count(self):
        # jackbot("******************* get_total_possess_count!!!")
        sql = "select count(code) from all_item_db where sell_date = '%s'"
        return self.engine_simulator.execute(sql % (0)).fetchall()[0][0]

    # jango_data 테이블을 만드는 함수
    def db_to_jango(self, date_rows_today):
        # 정산 함수
        self.check_balance()
        if self.is_simul_table_exist(self.db_name, "all_item_db") == False:
            return

        self.jango.loc[0, 'date'] = date_rows_today

        # self.jango.loc[0, 'total_asset'] = self.total_invest_price - self.loan_money
        self.jango.loc[0, 'today_profit'] = self.get_today_profit(date_rows_today)
        self.jango.loc[0, 'sum_valuation_profit'] = self.sum_valuation_profit
        self.jango.loc[0, 'total_profit'] = self.total_valuation_profit

        self.jango.loc[0, 'total_invest'] = self.total_invest_price
        self.jango.loc[0, 'd2_deposit'] = self.d2_deposit
        # 총매입금액
        self.jango.loc[0, 'sum_item_total_purchase'] = self.get_sum_item_total_purchase()

        # 총평가금액
        self.jango.loc[0, 'total_evaluation'] = self.get_sum_valuation_price()
        self.jango.loc[0, 'today_profitcut_count'] = self.get_today_profitcut_count(date_rows_today)
        self.jango.loc[0, 'today_losscut_count'] = self.get_today_losscut_count(date_rows_today)

        self.jango.loc[0, 'today_invest_price'] = float(self.today_invest_price)

        # self.jango.loc[0, 'today_reinvest_price'] = self.today_reinvest_price
        self.jango.loc[0, 'today_sell_price'] = self.get_sum_today_sell_price(date_rows_today)

        # 오늘 기준 수익률 (키움 잔고 상단에 뜨는 수익률) -0.33 (수수료, 세금)
        try:
            self.jango.loc[0, 'today_rate'] = round(
                (float(self.jango.loc[0, 'total_evaluation']) - float(
                    self.jango.loc[0, 'sum_item_total_purchase'])) / float(
                    self.jango.loc[0, 'sum_item_total_purchase']) * 100 - 0.33, 2)
        except ZeroDivisionError as e:
            pass

        # self.jango.loc[0, 'volume_limit'] = self.volume_limit

        # self.jango.loc[0, 'reinvest_point'] = self.reinvest_point
        self.jango.loc[0, 'sell_point'] = self.sell_point
        # self.jango.loc[0, 'max_reinvest_count'] = self.max_reinvest_count
        self.jango.loc[0, 'invest_limit_rate'] = self.invest_limit_rate
        self.jango.loc[0, 'invest_unit'] = self.invest_unit

        self.jango.loc[0, 'limit_money'] = self.limit_money
        self.jango.loc[0, 'total_possess_count'] = self.get_total_possess_count()
        self.jango.loc[0, 'today_buy_list_count'] = self.len_df_realtime_daily_buy_list
        # self.jango.loc[0, 'today_reinvest_count'] = self.get_today_reinvest_count(date_rows_today)
        # self.jango.loc[0, 'today_cant_reinvest_count'] = self.get_today_cant_reinvest_count()

        # 오늘 익절한 금액
        self.jango.loc[0, 'today_profitcut'] = self.get_sum_today_profitcut(date_rows_today)
        # 오늘 손절한 금액
        self.jango.loc[0, 'today_losscut'] = self.get_sum_today_losscut(date_rows_today)

        # 지금까지 총 익절한 금액
        self.jango.loc[0, 'total_profitcut'] = self.get_sum_total_profitcut()

        # 지금까지 총 손절한 금액
        self.jango.loc[0, 'total_losscut'] = self.get_sum_total_losscut()

        # 지금까지 총 익절한놈들
        self.jango.loc[0, 'total_profitcut_count'] = self.get_sum_total_profitcut_count()

        # 지금까지 총 손절한 놈들

        self.jango.loc[0, 'total_losscut_count'] = self.get_sum_total_losscut_count()

        self.jango.loc[0, 'today_buy_count'] = 0
        self.jango.loc[0, 'today_buy_total_sell_count'] = 0
        self.jango.loc[0, 'today_buy_total_possess_count'] = 0

        self.jango.loc[0, 'today_buy_today_profitcut_count'] = 0

        self.jango.loc[0, 'today_buy_today_losscut_count'] = 0
        self.jango.loc[0, 'today_buy_total_profitcut_count'] = 0

        self.jango.loc[0, 'today_buy_total_losscut_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count0_sell_count'] = 0
        #
        # self.jango.loc[0, 'today_buy_reinvest_count1_sell_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count2_sell_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count3_sell_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count4_sell_count'] = 0
        #
        # self.jango.loc[0, 'today_buy_reinvest_count4_sell_profitcut_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count4_sell_losscut_count'] = 0
        #
        # self.jango.loc[0, 'today_buy_reinvest_count5_sell_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count5_sell_profitcut_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count5_sell_losscut_count'] = 0
        #
        # self.jango.loc[0, 'today_buy_reinvest_count0_remain_count'] = 0
        #
        # self.jango.loc[0, 'today_buy_reinvest_count1_remain_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count2_remain_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count3_remain_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count4_remain_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count4_remain_count'] = 0
        # self.jango.loc[0, 'today_buy_reinvest_count5_remain_count'] = 0

        # # 데이터베이스에 테이블이 존재할 때 수행 동작을 지정한다.
        # 'fail', 'replace', 'append' 중 하나를 사용할 수 있는데 기본값은 'fail'이다.
        # 'fail'은 데이터베이스에 테이블이 있다면 아무 동작도 수행하지 않는다.
        # 'replace'는 테이블이 존재하면 기존 테이블을 삭제하고 새로 테이블을 생성한 후 데이터를 삽입한다.
        # 'append'는 테이블이 존재하면 데이터만을 추가한다.
        self.jango.to_sql('jango_data', self.engine_simulator, if_exists='append')

        #     # today_earning_rate
        sql = "update jango_data set today_earning_rate =round(today_profit / total_invest * '%s',2) WHERE date='%s'"
        # rows[i][0] 하는 이유는 rows[i]는 튜플( )로 나온다 그 튜플의 원소를 꺼내기 위해 rows[i]에 [0]을 추가
        self.engine_simulator.execute(sql % (100, date_rows_today))

    # 시뮬레이션이 다 끝났을 때 마지막 jango_data 정리
    def arrange_jango_data(self):
        if self.engine_simulator.dialect.has_table(self.engine_simulator, 'jango_data'):
            len_date = self.get_len_jango_data_date()
            sql = "select date from jango_data"
            rows = self.engine_simulator.execute(sql).fetchall()

            print('jango_data 최종 정산 중...')
            # 위에 전체
            for i in range(len_date):
                # today_buy_count
                sql = "UPDATE jango_data SET today_buy_count=(select count(*) from (select code from all_item_db where buy_date like '%s') b) WHERE date='%s'"
                # date 하는 이유는 rows[i]는 튜플로 나온다 그 튜플의 원소를 꺼내기 위해 [0]을 추가
                self.engine_simulator.execute(sql % ("%%" + str(rows[i][0]) + "%%", rows[i][0]))

                # today_buy_total_sell_count ( 익절, 손절 포함)
                sql = "UPDATE jango_data SET today_buy_total_sell_count=(select count(*) from (select code from all_item_db a where buy_date like '%s' and (a.sell_date != 0) group by code ) b) WHERE date='%s'"
                self.engine_simulator.execute(sql % ("%%" + rows[i][0] + "%%", rows[i][0]))

                # today_buy_total_possess_count 오늘 사고 계속 가지고 있는것들
                sql = "UPDATE jango_data SET today_buy_total_possess_count=(select count(*) from (select code from all_item_db a where buy_date like '%s' and a.sell_date = '%s' group by code ) b) WHERE date='%s'"
                self.engine_simulator.execute(sql % ("%%" + rows[i][0] + "%%", 0, rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_today_profitcut_count=(select count(*) from (select code from all_item_db where buy_date like '%s' and sell_date like '%s' and (sell_rate >= '%s' ) group by code ) b) WHERE date='%s'"
                self.engine_simulator.execute(sql % ("%%" + rows[i][0] + "%%", "%%" + rows[i][0] + "%%", 0, rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_today_profitcut_rate= round(today_buy_today_profitcut_count /today_buy_count *100,2) WHERE date = '%s'"
                self.engine_simulator.execute(sql % (rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_today_losscut_count=(select count(*) from (select code from all_item_db where buy_date like '%s' and sell_date like '%s' and sell_rate < '%s'  group by code ) b) WHERE date='%s'"
                self.engine_simulator.execute(sql % ("%%" + rows[i][0] + "%%", "%%" + rows[i][0] + "%%", 0, rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_today_losscut_rate=round(today_buy_today_losscut_count /today_buy_count *100,2) WHERE date = '%s'"
                self.engine_simulator.execute(sql % (rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_total_profitcut_count=(select count(*) from (select code from all_item_db where buy_date like '%s' and sell_rate >= '%s'  group by code ) b) WHERE date='%s'"
                self.engine_simulator.execute(sql % ("%%" + rows[i][0] + "%%", 0, rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_total_profitcut_rate=round(today_buy_total_profitcut_count /today_buy_count *100,2) WHERE date = '%s'"
                self.engine_simulator.execute(sql % (rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_total_losscut_count=(select count(*) from (select code from all_item_db where buy_date like '%s' and sell_rate < '%s'  group by code ) b) WHERE date='%s'"
                self.engine_simulator.execute(sql % ("%%" + rows[i][0] + "%%", 0, rows[i][0]))

                sql = "UPDATE jango_data SET today_buy_total_losscut_rate=round(today_buy_total_losscut_count/today_buy_count*100,2) WHERE date = '%s'"
                self.engine_simulator.execute(sql % (rows[i][0]))
        print('jango_data 최종 정산 완료')

    # 분 데이터를 가져오는 함수
    def get_date_min_for_simul(self, simul_start_date):
        
        dt_format = '%Y%m%d%H%M'
        simul_time = datetime.datetime.strptime(simul_start_date + "0900", dt_format)
        min_delta = datetime.timedelta(minutes=1)

        times = []
        while simul_time.hour != 15 or simul_time.minute != 31:
            times.append((datetime.datetime.strftime(simul_time, dt_format),))
            simul_time += min_delta

        self.min_date_rows = times
    # 분별 시뮬레이팅 함수
    # 새로운 종목 매수 및 보유한 종목의 데이터를 업데이트 하는 함수, 매도 함수도 포함
    def trading_by_min(self, date_rows_today, date_rows_yesterday, i):
        self.print_info(date_rows_today)

        # all_item_db가 존재하고, 현재 보유 중인 종목이 있다면 아래 로직을 들어간다.
        if self.is_simul_table_exist(self.db_name, "all_item_db") and len(self.get_data_from_possessed_item()) != 0:
            # 보유 중인 종목들의 주가를 일별로 업데이트 하는 함수(option 이 OPEN 이면 OPEN가만 업데이트)
            self.update_all_db_by_date(date_rows_today, option='OPEN')

        # 분별 시간 데이터를 가져온다.
        self.get_date_min_for_simul(date_rows_today)
        if len(self.min_date_rows) != 0:
            # 분 단위로 for문을 돈다
            for t in range(len(self.min_date_rows)):
                min = self.min_date_rows[t][0]
                # all_item_db가 존재하고 현재 보유 중인 종목이 있는 경우
                if self.is_simul_table_exist(self.db_name,"all_item_db") and len(self.get_data_from_possessed_item()) != 0:
                    self.print_info(min)
                    self.update_all_db_by_min(min)
                    self.update_all_db_etc()
                    # 매도 함수
                    self.auto_trade_sell_stock(min, i)
                    # self.buy_stop 이 False 이고, 보유 자산이 있으면 실제 매수를 한다.
                    if not self.buy_stop and self.jango_check():
                        # 매수 할 종목을 가져온다
                        self.get_realtime_daily_buy_list()

                        if self.len_df_realtime_daily_buy_list > 0:

                            self.auto_trade_stock_realtime(min, date_rows_today, date_rows_yesterday)
                        else:
                            print("realtime_daily_buy_list에 금일 매수 대상 종목이 0개 이다.  ")


                #  여긴 가장 초반에 all_itme_db를 만들어야 할때이거나 매수한 종목이 없을 때 들어가는 로직
                else:
                    if not self.buy_stop and self.jango_check():
                        self.auto_trade_stock_realtime(min, date_rows_today, date_rows_yesterday)

                # 9시에만 매수를 하는 경우는 한번만 9시에 매수 하고 self.buy_stop을 true로 변경하여 이후로 매수하지 않도록 설정
                if not self.buy_stop and self.only_nine_buy:
                    print("9시 매수 끝!!!!!!!!!!")
                    self.buy_stop = True


        else:
            print("min_craw db의 종목 테이블에 " + str(
                date_rows_today) + " 데이터가 존재 하지 않는다! self.simul_start_date 날짜를 변경 하세요! (분별 데이터는 콜렉터에서 최근 1년 데이터만 가져옵니다! ")

    # 새로운 종목 매수 및 보유한 종목의 데이터를 업데이트 하는 함수, 매도 함수도 포함
    def trading_by_date(self, date_rows_today, date_rows_yesterday, i):
        self.print_info(date_rows_today)

        # all_item_db가 존재하고, 현재 보유 중인 종목이 있다면 아래 로직을 들어간다.
        if self.is_simul_table_exist(self.db_name, "all_item_db") and len(self.get_data_from_possessed_item()) != 0:
            # 보유 중인 종목들의 주가를 일별로 업데이트 하는 함수
            self.update_all_db_by_date(date_rows_today, option = 'OPEN')
            # 보유 중인 종목들의 주가 이외의 기타 정보들을 업데이트 하는 함수
            self.update_all_db_etc()
            # 매도 함수
            self.auto_trade_sell_stock(date_rows_today, i)

            # 보유 자산이 있다면, 실제 매수를 한다.
            if self.jango_check():
                # 돈있으면 매수 시작
                self.auto_trade_stock_realtime(str(date_rows_today) + "0900", date_rows_today, date_rows_yesterday)

        #  여긴 가장 초반에 all_itme_db를 만들어야 할때이거나 매수한 종목이 없을 때 들어가는 로직
        else:
            if self.jango_check():
                self.auto_trade_stock_realtime(str(date_rows_today) + "0900", date_rows_today, date_rows_yesterday)

    # 매일 시뮬레이팅 돌기 전 초기화 세팅
    def daily_variable_setting(self):
        self.buy_stop = False
        self.today_invest_price = 0

    # 분별 시뮬레이팅
    def simul_by_min(self, date_rows_today, date_rows_yesterday, i):
        print("**************************   date: " + date_rows_today)
        # 일별 시뮬레이팅 하며 변수 초기화(분별시뮬레이터의 경우도 하루 단위로 초기화)
        self.daily_variable_setting()
        # daily_buy_list에 시뮬레이팅 할 날짜에 해당하는 테이블과 전 날 테이블이 존재하는지 확인
        if self.is_date_exist(date_rows_today) and self.is_date_exist(date_rows_yesterday):
            # 우선 매수리스트를 가져온다.
            self.db_to_realtime_daily_buy_list(date_rows_today, date_rows_yesterday, i)
            # 분별 시뮬레이팅 시작한다.
            self.trading_by_min(date_rows_today, date_rows_yesterday, i)
            self.db_to_jango(date_rows_today)

            # [추가 코드]all_item_db가 존재하고, 현재 보유 중인 종목이 있다면 아래 로직을 들어간다.
            if self.is_simul_table_exist(self.db_name, "all_item_db") and len(self.get_data_from_possessed_item()) != 0:
                # 보유 중인 종목들의 주가를 일별로 업데이트 하는 함수(분별 종가 업데이트 이외에 clo5, clo20등등의 값을 업데이트)
                self.update_all_db_by_date(date_rows_today, option='ALL')

        else:
            print(date_rows_today + "테이블은 존재하지 않는다!!!")

    # 일별 시뮬레이팅
    def simul_by_date(self, date_rows_today, date_rows_yesterday, i):
        print("**************************   date: " + date_rows_today)
        # 일별 시뮬레이팅 하며 변수 초기화
        self.daily_variable_setting()
        # daily_buy_list에 시뮬레이팅 할 날짜에 해당하는 테이블과 전 날 테이블이 존재하는지 확인
        if self.is_date_exist(date_rows_today) and self.is_date_exist(date_rows_yesterday):
            # 당일 매수 할 종목들을 realtime_daily_buy_list 테이블에 세팅
            self.db_to_realtime_daily_buy_list(date_rows_today, date_rows_yesterday, i)
            # 트레이딩(매수, 매도) 함수 + 보유 종목의 현재가 업데이트 함수
            self.trading_by_date(date_rows_today, date_rows_yesterday, i)

            # [추가 코드]all_item_db가 존재하고, 현재 보유 중인 종목이 있다면 아래 로직을 들어간다.
            if self.is_simul_table_exist(self.db_name, "all_item_db") and len(self.get_data_from_possessed_item()) != 0:
                # 보유 중인 종목들의 주가를 일별로 업데이트 하는 함수(분별 종가 업데이트 이외에 clo5, clo20등등의 값을 업데이트)
                self.update_all_db_by_date(date_rows_today, option='ALL')

            # 일별 정산
            self.db_to_jango(date_rows_today)

        else:
            print(date_rows_today + "테이블은 존재하지 않는다!!!")

    # 날짜 별 로테이팅 함수
    def rotate_date(self):
        for i in range(1, len(self.date_rows)):
            # print("self.date_rows!!" ,self.date_rows)
            # 시뮬레이팅 할 일자
            date_rows_today = self.date_rows[i][0]
            # 시뮬레이팅 하기 전의 일자
            date_rows_yesterday = self.date_rows[i - 1][0]

            # self.simul_reset 이 False, 즉 시뮬레이터를 멈춘 지점 부터 실행하기 위한 조건 
            if not self.simul_reset and not self.simul_reset_lock:
                if int(date_rows_today) <= int(self.last_simul_date):
                    print("**************************   date: " + date_rows_today + "simul jango date exist pass ! ")
                    continue
                else:
                    self.simul_reset_lock = True

            # 분별 시뮬레이팅
            if self.use_min:
                self.simul_by_min(date_rows_today, date_rows_yesterday, i)
            # 일별 시뮬레이팅
            else:
                self.simul_by_date(date_rows_today, date_rows_yesterday, i)

        # 마지막 jango_data 정리
        self.arrange_jango_data()


# 
def escape_percentage(conn, clauseelement, multiparams, params):
    # execute로 실행한 sql문이 들어왔을 때 %를 %%로 replace
    if isinstance(clauseelement, str) and '%' in clauseelement and multiparams is not None:
        while True:
            replaced = re.sub(r'([^%])%([^%s])', r'\1%%\2', clauseelement)
            if replaced == clauseelement:
                break
            clauseelement = replaced

    return clauseelement, multiparams, params




# sell_AI 기능


def sell_list_ai(dataset, sell_ai_settings):
    """
    :param dataset: 실제 주가 데이터
    :param settings: AI 알고리즘 세팅
    :return
    """

    shuffled_data = load_data(df=dataset.copy(), n_steps=sell_ai_settings['n_steps'], test_size=sell_ai_settings['test_size'])
    
    #!@

    #model = create_model(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                     n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    # model = create_model_Bidirectional(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                         n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    
        
    #model = CNN_Attention_BiLSTM_Version17()                        
    model = sell_ai_settings['model'] 
    
    #checkpoint_filepath = 'ModelCheckpoint/CNN_Attention_BiLSTM_Version3/Checkpoint'

    #early_stopping = EarlyStopping(monitor='val_loss', patience=500)  # patience 번이상 더 좋은 결과가 없으면 학습을 멈춤
    #callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
    #                                          monitor='val_loss', 
    #                                          save_best_only=True, verbose=1)
    
    #ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, verbose=1, mode='min',monitor='val_loss')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                           patience=5, min_lr=0.001, mode='min',verbose=1)
    #wandb.init(project="simulation_Sell", entity="SeongJae-Yoo")
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
                        batch_size=sell_ai_settings['batch_size'],
                        epochs=sell_ai_settings['epochs'],
                        validation_data=(shuffled_data["X_test"], shuffled_data["y_test"]),
                        callbacks=[reduce_lr],
                        verbose=1)

    scaled_data = load_data(df=dataset.copy(), n_steps=sell_ai_settings['n_steps'], test_size=sell_ai_settings['test_size'],
                            shuffle=False)

    # result = evaluate(scaled_data, model)
    # print(f"result: {result}")

    
    # train_huber_loss, train_mae, train_rmse,test_huber_loss, test_mae, test_rmse = evaluate(scaled_data, model)  
    # print(f"train_huber_loss, train_mae, train_rmse: {train_huber_loss,train_mae, train_rmse}")      
    # print(f"test_huber_loss, test_mae, test_rmse: {test_huber_loss,test_mae, test_rmse}")      

    # mse = evaluate(scaled_data, model)
    # print(f"Mean Squared Error: {mse}")



    # 예측 가격
    future_price = predict(scaled_data, model, n_steps=sell_ai_settings['n_steps'])

    # 스케일링 된 예측 결과
    scaled_y_pred = model.predict(scaled_data['X_test'])
    # 실제 값으로 변환 된 결과
    y_pred = np.squeeze(scaled_data['column_scaler']['close'].inverse_transform(scaled_y_pred))

    if sell_ai_settings['is_used_predicted_close']:
        close = y_pred[-1] # 예측 그래프에서의 종가
    else:
        close = dataset.iloc[-1]['close'] # 실제 종가

    # ratio : 예상 상승률
    ratio = (future_price - close) / close * 100

    msg = f"After {sell_ai_settings['lookup_step']}: {int(close)} -> {int(future_price)}"

    if ratio > 0: # lookup_step(분, 일) 후 상승 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯅ '
    elif ratio < 0: # lookup_step(분, 일) 후 하락 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯆ '
    print(msg, end=' ')
    return sell_ai_settings['ratio_cut'] > ratio # ratio_cut(목표 수익률) 보다 ratio가 작으면 True 반환(필터링 대상)
# -2>= -3 , 

def sell_list_ai_v2(dataset, sell_ai_settings):
    """
    :param dataset: 실제 주가 데이터
    :param settings: AI 알고리즘 세팅
    :return
    """

    shuffled_data = load_data(df=dataset.copy(), n_steps=sell_ai_settings['n_steps'], test_size=sell_ai_settings['test_size'])
    
    #!@

    #model = create_model(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                     n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    # model = create_model_Bidirectional(n_steps=ai_settings['n_steps'], loss=ai_settings['loss'], units=ai_settings['units'],
    #                         n_layers=ai_settings['n_layers'], dropout=ai_settings['dropout'])
    
        
    #model = CNN_Attention_BiLSTM_Version9()                      
    model = sell_ai_settings['model'] 
    
    #checkpoint_filepath = 'ModelCheckpoint/CNN_Attention_BiLSTM_Version3/Checkpoint'

    #early_stopping = EarlyStopping(monitor='val_loss', patience=500)  # patience 번이상 더 좋은 결과가 없으면 학습을 멈춤
    #callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
    #                                          monitor='val_loss', 
    #                                          save_best_only=True, verbose=1)
    
    #ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, verbose=1, mode='min',monitor='val_loss')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                           patience=5, min_lr=0.001, mode='min',verbose=1)
    #wandb.init(project="simulation_Sell", entity="SeongJae-Yoo")
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
    # wandb.log({"gradients": wandb.Histogram(numpy_array_or_sequence)})
    # wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})

    # wandb   Hyperparameter Sweeps   시각화 할 수 있는 자료 (아래 확인)
    #https://colab.research.google.com/drive/1gKixa6hNUB8qrn1CfHirOfTEQm0qLCSS#scrollTo=1gD9qhA9yOYs
    
    #wandb_callback = WandbCallback(monitor='val_loss',save_model=True,mode='min',log_weights=True,log_evaluation=True,validation_steps=5,verbose=1)
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.0001, mode='min',verbose=1)
    
    model.fit(shuffled_data["X_train"], shuffled_data["y_train"],
                        batch_size=sell_ai_settings['batch_size'],
                        epochs=sell_ai_settings['epochs'],
                        validation_data=(shuffled_data["X_test"], shuffled_data["y_test"]),
                        callbacks=[reduce_lr],
                        verbose=1)

    scaled_data = load_data(df=dataset.copy(), n_steps=sell_ai_settings['n_steps'], test_size=sell_ai_settings['test_size'],
                            shuffle=False)

    # result = evaluate(scaled_data, model)
    # print(f"result: {result}")

    
    # train_huber_loss, train_mae, train_rmse,test_huber_loss, test_mae, test_rmse = evaluate(scaled_data, model)  
    # print(f"train_huber_loss, train_mae, train_rmse: {train_huber_loss,train_mae, train_rmse}")      
    # print(f"test_huber_loss, test_mae, test_rmse: {test_huber_loss,test_mae, test_rmse}")      

    # mse = evaluate(scaled_data, model)
    # print(f"Mean Squared Error: {mse}")



    # 예측 가격
    future_price = predict(scaled_data, model, n_steps=sell_ai_settings['n_steps'])

    # 스케일링 된 예측 결과
    scaled_y_pred = model.predict(scaled_data['X_test'])
    # 실제 값으로 변환 된 결과
    y_pred = np.squeeze(scaled_data['column_scaler']['close'].inverse_transform(scaled_y_pred))

    if sell_ai_settings['is_used_predicted_close']:
        close = y_pred[-1] # 예측 그래프에서의 종가
    else:
        close = dataset.iloc[-1]['close'] # 실제 종가

    # ratio : 예상 상승률
    ratio = (future_price - close) / close * 100

    msg = f"After {sell_ai_settings['lookup_step']}: {int(close)} -> {int(future_price)}"

    if ratio > 0: # lookup_step(분, 일) 후 상승 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯅ '
    elif ratio < 0: # lookup_step(분, 일) 후 하락 예상일 경우 출력 메시지
        msg += f'    {ratio:.2f}% ⯆ '
    print(msg, end=' ')
    return sell_ai_settings['ratio_cut'] > ratio # ratio_cut(목표 수익률) 보다 ratio가 작으면 True 반환(필터링 대상)
# -2>= -3 ,2 


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



# ADD 11-29
def SMA(data,period = 30) :
    return data[:].rolling(window = period).mean()

def RSI(data, period = 14) :
    delta = data[:].diff(1)
    delta = delta.dropna()
    
    up = delta.copy()
    down = delta.copy()
    up[up <0] =0
    down[down>0] = 0
    data['up'] = up
    data['down'] = down
    
    AVG_Gain = SMA(data, period)
    AVG_Loss = abs(SMA(data,period))
    RS = AVG_Gain / AVG_Loss
    
    RSI = 100.0 - (100.0 / (1.0+RS))
    data[0] = RSI
    
    return data[0]


if __name__ == '__main__':
    logger.error('simulator.py로 실행해 주시기 바랍니다.')
    sys.exit(1)
