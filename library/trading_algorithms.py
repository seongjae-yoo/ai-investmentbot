import numpy as np
import pandas as pd

#ma_period = 70 
# result = BBands(pd.DataFrame(df_close), w=ma_period)
def BBands(df_close, w=70, k=3):
    """
        w: 이동평균선 기간 값 (20)
        k: 승수 (2)
        std 함수는 '표준편차를 구하기 위한' numpy 패키지에 포함되어 있는 내장 함수입니다.
        DATAFRAME[-1:] 마지막 row
        DATAFRAME[:-1] index 0부터 마지막 row 제외한 rows
        DATAFRAME[-20:] 뒤에서부터 20개의 rows
        DATAFRAME[:20] index 0부터 20개의 rows
        DATAFRAME[20:] index 20부터 끝까지 rows
    """
    # 고가, 저가, 종가의 평균을 이용하는 경우 정수로 변환이 필요
    df_close = df_close.astype(int)
    # 표준편차
    std = df_close[:w].std()[0]
    # mean() 함수는 '평균을 구하기 위한' numpy 패키지에 포함되어 있는 내장 함수입니다.
    # 70일 이평선이자 볼린저밴드 중앙선
    mbb = df_close[:w].mean()[0]
    # 종가
    close = df_close[0][0]

    '''
        std (표준편차 값)과 mbb(중앙선)을 이용하여 볼린저밴드
        1. ubb (상한선)
        2. lbb (하한선)
        3. perb (%b: 볼린저밴드에서의 종가 위치)
        4. bw (밴드폭)
        mbb = 중심선 = 주가의 20 기간 이동평균선 = clo20
        ubb = 상한선 = 중심선 + 주가의 20기간 표준편차 * 2 
        lbb = 하한선 = 중심선 – 주가의 20기간 표준편차 * 2  
        perb = %b = (주가 – 하한선) / (상한선 – 하한선) = (close - lbb) / (ubb - lbb)
        bw = 밴드폭 (Bandwidth) = (상한선 – 하한선) / 중심선 = (ubb - lbb) / mbb

        *  참고 사이트 - https://humankind.tistory.com/33            
        %B quantifies a security's price relative to the upper and lower Bollinger Band. There are six basic relationship levels:
        %B is below 0 when price is below the lower band
        %B equals 0 when price is at the lower band
        %B is between 0 and .50 when price is between the lower and middle band (20-day SMA)
        %B is between .50 and 1 when price is between the upper and middle band (20-day SMA)
        %B equals 1 when price is at the upper band
        %B is above 1 when price is above the upper band



    '''
    # 볼린저 밴드의 상한선: 70일 이평선 값 + (70일 동안의 주가 표준편차 값 ) * 2
    # 볼린저 밴드의 하한선: 70일 이평선 값 - ( 70일 동안의 주가 표준편차 값 ) * 2
    # ubb = mbb + (std * 2)
    # lbb =  mbb - (std * 2)
    # Calculate the upper and lower Bollinger Bands
# upper_band = MA + 2 * STD
# lower_band = MA - 2 * STD

    #1.8 
    ubb = mbb + std * 3
    lbb =  mbb - std * 3

    if ubb > lbb:
        perb = (close - lbb) / (ubb - lbb)
        bw =  (ubb - lbb) / mbb
        return mbb, ubb, lbb, perb, bw
    else:
        return False
