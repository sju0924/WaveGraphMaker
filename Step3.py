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
    
    for i in range(100):
        try:
            idx = str(i+1)
            eq_filename_x = path + "eq"+idx+"x.eq"
            eq_filename_y = path + "eq"+idx+"y.eq"

            with open(eq_filename_x,'w',encoding='utf-8') as f:
                for item in df[idx+"xAcc"]:
                    if item == 'NaN':
                        break
                    f.write(str(item).upper() + '\n')

            with open(eq_filename_y,'w',encoding='utf-8') as f:
                for item in df[idx+"yAcc"]:
                    if item == 'NaN':
                        break
                    f.write(str(item).upper() + '\n')     
        except KeyError:
            break
        
def get_srss_table():
    
    # Create a DataFrame
    df = pd.DataFrame([], columns=['T'])

    # 응답스펙트럼 X,Y 방향 테이블 불러오기
    path = PATH_STEP3 + "/input/SGS_result"
    for i in range(100):
        idx = str(i+1)
        eq_filename_x = path + "/"+idx+"x.sgs"
        eq_filename_y = path + "/"+idx+"y.sgs"       
        
        try:
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
        except FileNotFoundError:
            break

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
    for i in range(100):
        idx = str(i+1)    
        acc_filename_x = path + "/"+idx+"x.txt"
        acc_filename_y = path + "/"+idx+"y.txt"       
        try:
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
        except FileNotFoundError:
            break
        except ValueError as e:
            print(e)
            print(acc_filename_x)
            break

    return df

def Step3_eq():
    generate_eq()

def Step3_srss():
    try:
        scale = get_scaleup_factor()
    except FileNotFoundError:
        print("Scale up Factor 파일이 존재하지 않습니다. Step 2에서 생성해주세요")
        return
    if len(scale) == 0:
        print("Scale up Factor 파일이 손상되었습니다.")
        return
    srss_table = get_srss_table()
    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP3 + "/output/SRSS_Scale",mod='total')

def Step3_acc():
    try:
        scale = get_scaleup_factor()
    except FileNotFoundError:
        print("Scale up Factor 파일이 존재하지 않습니다. Step 2에서 생성해주세요")
        return
    if len(scale) == 0:
        print("Scale up Factor 파일이 손상되었습니다.")
        return
    
    acc_table = get_acceleration_table()
    plot_acceleration(acc_table, scale, PATH_STEP3 + "/output/Acceleration")

def main():

    # eq 파일 만들기
    generate_eq()

    cmd = ""
    
    while(cmd != "yes"):
        print("SGS 결과값을 input 폴더에 넣어주세요")
        cmd = input("완료 후 'yes' 를 입력하세요 > ")
    
    scale = get_scaleup_factor()
    
    acc_table = get_acceleration_table()
    plot_acceleration(acc_table, scale, PATH_STEP3 + "/output/Acceleration")

    cmd = ""
    while(cmd != "yes"):
        print("Shake_M 결과값을 input 폴더에 넣어주세요")
        cmd = input("완료 후 'yes' 를 입력하세요 > ")

    srss_table = get_srss_table()
    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP3 + "/output/SRSS_Scale")
    

    
if __name__=="__main__":
    main()

