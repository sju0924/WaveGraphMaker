import pandas as pd
import os
import numpy as np
from math import sqrt
from Step1 import get_scaleup_factor
from Step2 import plot_response, plot_scaleup_factor, plot_acceleration

PATH_STEP4 = "./Step4"

def get_srss_table():
    
    # Create a DataFrame
    df = pd.DataFrame([], columns=['T'])

    # 응답스펙트럼 X,Y 방향 테이블 불러오기
    path = PATH_STEP4 + "/input/SGS_result"
    for i in range(7):
        idx = str(i+1)
        eq_filename_x = path + "/"+idx+"x.sgs"
        eq_filename_y = path + "/"+idx+"y.sgs"       
        
        with open(eq_filename_x,'r',encoding='utf-8') as f:

            lines = f.read().splitlines()[9:-1]

            # x방향 데이터 불러오기기
            numeric_data = [line.split(',') for line in lines]
            df_x = pd.DataFrame(numeric_data, columns=['T', idx + 'x'])

            if len(numeric_data) > len(df['T']):
                df['T'] = pd.to_numeric(df_x['T'])

            df[idx + 'x'] = pd.to_numeric(df_x[idx + 'x'])


        with open(eq_filename_y,'r',encoding='utf-8') as f:
            
            lines = f.read().splitlines()[9:-1]

            # x방향 데이터 불러오기기
            numeric_data = [line.split(',') for line in lines]
            df_y = pd.DataFrame(numeric_data, columns=['T', idx + 'y'])

            if len(numeric_data) > len(df['T']):
                df['T'] = pd.to_numeric(df_y['T'])  

            df[idx + 'y'] = pd.to_numeric(df_y[idx + 'y'])
        
        # srss 계산산
        df[str(i+1)+'srss'] = df[str(i+1)+'x'] * df[str(i+1)+'x'] + df[str(i+1)+'y'] * df[str(i+1)+'y']
        df[str(i+1)+'srss'] = df[str(i+1)+'srss'].apply(sqrt)      

    df.set_index('T', inplace=True)

    return df

# @input    srss_table: x,y, srss 컬럼을 포함한 데이터프레임
# @output   Step2/SRSS 위치에 srss table의 그래프 이미지 파일 저장
def plot_srss(srss_table):
    plot_response(srss_table, PATH_STEP4 + "/output/SRSS")


def get_acceleration_table():
    # Create a DataFrame
    df = pd.DataFrame()

    # 응답스펙트럼 X,Y 방향 테이블 불러오기
    path = PATH_STEP4 + "/input/RSP_match_result"
    for i in range(7):
        idx = str(i+1)    
        acc_filename_x = path + "/"+idx+"x.txt"
        acc_filename_y = path + "/"+idx+"y.txt"       
        
        with open(acc_filename_x,'r',encoding='utf-8') as f:
            
            lines = f.read().splitlines()

            # x방향 데이터 불러오기기
            numeric_data = [line.split() for line in lines]  
            metadata = numeric_data[1]
            npts = int(metadata[0])  # 데이터 포인트 수
            delta_t = float(metadata[1])  # 샘플링 간격

            # 각 열의 데이터를 배열에 추가  
            data = []
            for line in numeric_data[2:]:  # 2번째 줄부터 데이터 시작
                data.extend(map(float, line.split()))       
       
            # NumPy 배열로 변환
            acceleration = np.array(data)

            # 시간 배열 생성
            time = np.arange(0, npts * delta_t, delta_t)

            # DataFrame에 추가가
            df[idx +'xT'] = time
            df[idx + 'xAcc'] = acceleration

        with open(acc_filename_y,'r',encoding='utf-8') as f:
            
            lines = f.read().splitlines()

            # y방향 데이터 불러오기기
            numeric_data = [line.split() for line in lines]  
            metadata = numeric_data[1]
            npts = int(metadata[0])  # 데이터 포인트 수
            delta_t = float(metadata[1])  # 샘플링 간격

            # 각 열의 데이터를 배열에 추가  
            data = []
            for line in numeric_data[2:]:  # 2번째 줄부터 데이터 시작
                data.extend(map(float, line.split()))       
       
            # NumPy 배열로 변환
            acceleration = np.array(data)

            # 시간 배열 생성
            time = np.arange(0, npts * delta_t, delta_t)

            # DataFrame에 추가
            df[idx +'yT'] = time
            df[idx + 'yAcc'] = acceleration

    return df


def main():
    srss_table = get_srss_table()
    scale = get_scaleup_factor(srss_table, new = 1)
    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP4 + "/output/SRSS_Scale")

    acc_table = get_acceleration_table()
    plot_acceleration(acc_table, scale, PATH_STEP4 + "/Acceleration")

    # ToDo: 1400년 주기 지진파 데이터 작성하기