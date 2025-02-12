import pandas as pd
import numpy as np
import os
from math import sqrt
from Step2 import plot_response, plot_scaleup_factor, plot_acceleration

PATH_STEP3 = "./Step3"
FILE_NAME = "./waves.xlsx"
def generate_eq():
    dataset = pd.read_excel(FILE_NAME, sheet_name = 'acc')
    df = pd.DataFrame(dataset)

    path = PATH_STEP3 + "/output/eq/"
    # 그래프 이미지 저장 폴더 생성
    if not os.path.exists(path):
        os.makedirs(path)

    # eq 파일 생성
    for i in range(7):
        idx = str(i+1)
        eq_filename_x = path + "eq"+idx+"x.eq"
        eq_filename_y = path + "eq"+idx+"y.eq"

        with open(eq_filename_x,'w',encoding='utf-8') as f:
            for item in df[idx+"xAcc"]:
                f.write(str(item).upper() + '\n')

        with open(eq_filename_y,'w',encoding='utf-8') as f:
            for item in df[idx+"yAcc"]:
                f.write(str(item).upper() + '\n')        
        
def get_srss_table():
    
    # Create a DataFrame
    df = pd.DataFrame([], columns=['T'])

    # 응답스펙트럼 X,Y 방향 테이블 불러오기
    path = PATH_STEP3 + "/input/SGS_result"
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
# @output   Step2/SRSS 위치에 srss table의 그래프 이미지 파일 저장장
def plot_srss(srss_table):
    plot_response(srss_table, PATH_STEP3+"/output/SRSS")

def get_scaleup_factor():
    with open("./scale_up_factor.txt", 'r') as sf:
        scale = [float (i) for i in sf.read().split(', ')]

    return scale

def get_acceleration_table():
    # Create a DataFrame
    df = pd.DataFrame()

    # 응답스펙트럼 X,Y 방향 테이블 불러오기
    path = PATH_STEP3 + "/input/Shake_M_result"
    for i in range(7):
        idx = str(i+1)    
        acc_filename_x = path + "/"+idx+"x.txt"
        acc_filename_y = path + "/"+idx+"y.txt"       
        
        with open(acc_filename_x,'r',encoding='utf-8') as f:
            
            lines = f.read().splitlines()

            # x방향 데이터 불러오기기
            numeric_data = [line.split() for line in lines]            
       
            df_x = pd.DataFrame(numeric_data, columns=['T', idx+'x_raw', idx + 'x', 'unknown'])
           
            df[idx + 'xT'] = pd.to_numeric(df_x['T'])
            df[idx + 'xAcc'] = pd.to_numeric(df_x[idx + 'x'])

        with open(acc_filename_y,'r',encoding='utf-8') as f:
            
            lines = f.read().splitlines()

            # x방향 데이터 불러오기기
            numeric_data = [line.split() for line in lines]

            df_y = pd.DataFrame(numeric_data, columns=['T', idx+'y_raw', idx + 'y', 'unknown'])  
            df[idx + 'yT'] = pd.to_numeric(df_y['T'])
            df[idx + 'yAcc'] = pd.to_numeric(df_y[idx + 'y'])

    return df

# shakeM 프로그램을 실행하여 가속도 데이터 생성
def execute_shakeM():
    pass

# SGS 프로그램을 사용하여 10초 주기의 응
def execute_sgs():
    

    # Sample data
    time = np.linspace(0, 1, 1000)  # 1초 동안 1000개의 샘플
    acceleration = np.sin(2 * np.pi * 50 * time) + 0.5 * np.sin(2 * np.pi * 120 * time)

    # FFT 계산
    fft_result = np.fft.fft(acceleration)
    frequencies = np.fft.fftfreq(len(time), d=(time[1] - time[0]))

    # 양수 주파수만 사용
    positive_frequencies = frequencies[frequencies >= 0]
    magnitude = np.abs(fft_result[frequencies >= 0])


def main():

    # eq 파일 만들기
    generate_eq()

    cmd = ""
    
    while(cmd != "yes"):
        print("SGS 및 Shake_M 결과값을 input 폴더에 넣어주세요")
        cmd = input("완료 후 'yes' 를 입력하세요 > ")

    srss_table = get_srss_table()
    scale = get_scaleup_factor()
    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP3 + "/output/SRSS_Scale")
    
    acc_table = get_acceleration_table()
    plot_acceleration(acc_table, scale, PATH_STEP3 + "/output/Acceleration")
    
if __name__=="__main__":
    main()

