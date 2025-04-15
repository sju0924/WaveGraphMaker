from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from Step1 import get_response_spectrum, transform_x

FILE_NAME= "./waves.xlsx"
PATH_STEP2 = "./Step2"
# 손실 함수 정의 (L2 노름)
def loss_function(scale, X, Y,weights):
    scaled_X = X * scale
    return np.sum(weights * (Y - scaled_X) ** 2)  # L2 노름 계산

def get_srss_table():
    dataset = pd.read_excel(FILE_NAME, sheet_name = 'data')
    df = pd.DataFrame(dataset)
    df.set_index('T', inplace=True)
    
    for i in range(100):
        try:
            df[str(i+1)+'srss'] = df[str(i+1)+'x'] * df[str(i+1)+'x'] + df[str(i+1)+'y'] * df[str(i+1)+'y']
            df[str(i+1)+'srss'] = df[str(i+1)+'srss'].apply(sqrt)
        except KeyError:
            break

    return df
# @input    table: 응답 스펙트럼 데이터 테이블. x,y 및 srss 데이터를 포함
#           path: 그래프를 저장하기 위한 경로
#           mod: general: 각 srss 그래프만 생성 total: 전체 srss 그래프를 하나로 통합하는 그래프를 추가로 생성성
# @output   path 위치에 그래프 이미지 파일을 저장
def plot_response(table, path, mod='general'):
    
    # 기존 DataFrame의 인덱스를 변환된 x축 값으로 설정
    with open("config.json", "r") as json_file:
        config = json.load(json_file)
    T_min = config["T_min"]
    T_max = config["T_max"]
    ratio_tmin = config["ratio_tmin"]
    ratio_second_half = config["ratio_second_half"] = {int(k): v for k, v in config["ratio_second_half"].items()}

    df_response = get_response_spectrum()
    df_response_transform = df_response.copy()
    df_response_transform.index = df_response_transform["T"].map(lambda x: transform_x(x, T_min*0.2, ratio_tmin, ratio_second_half))


    df_transform = table.copy()
    df_transform.index = df_transform.index.map(lambda x: transform_x(x, T_min*0.2, ratio_tmin, ratio_second_half))


    # 그래프 이미지 저장 폴더 생성
    if not os.path.exists(path):
        os.makedirs(path)

    # SRSS 그래프 그리기
    for i in range(100):
        try:
            idx = str(i+1)
                    
            plt.figure(figsize=(11, 6))

            plt.plot(df_response_transform.index, df_response_transform["1.3*MCE*0.8"], label="1.3*MCE*0.8")
            plt.plot(df_transform.index, df_transform[idx+"x"], label="X-Dir")
            plt.plot(df_transform.index, df_transform[idx+"y"], label="Y-Dir")
            plt.plot(df_transform.index, df_transform[idx+"srss"], label="SRSS")
            

            xticks = [0]
            xticks_label=['0']
            for key, value in config["ratio_second_half"].items():
                
                if key == 1 or key == 10:
                    xticks_label.append(str(key))
                    xticks.append(value)
                elif key == 11:
                    pass
                else:
                    xticks_label.append("")
                    xticks.append(value)

                
            # x축 범위 매칭
            plt.xticks(xticks, xticks_label)         
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            # 그래프 꾸미기
            plt.title("EQ" + idx)
            plt.xlabel("Period")
            plt.ylabel("Spectral Acceleration")
            plt.legend()
            plt.grid(True)
        
            # 그래프 출력
            plt.savefig(path + "/EQ" + idx + ".png")
            plt.clf()
        except KeyError:
            break
    
    if mod == 'total':
        plt.figure(figsize=(11, 6))
        plt.plot(df_response_transform.index, df_response_transform["1.3*MCE*0.8"], label="1.3*MCE*0.8")
        plt.plot(df_response_transform.index, df_response_transform["1.3*MCE*0.9"], label="1.3*MCE*0.9")

        for i in range(100):
            
            try:
                idx = str(i+1)                          
                plt.plot(df_transform.index, df_transform[idx+"srss"], label="EQ"+idx)

            except KeyError:
                break
        # x축 범위 매칭
        plt.xticks(xticks, xticks_label)         
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        # 그래프 꾸미기
        plt.title("SRSS")
        plt.xlabel("Period")
        plt.ylabel("Spectral Acceleration")
        plt.legend()
        plt.grid(True)
    
        # 그래프 출력
        plt.savefig(path + "/SRSS.png")
        plt.clf()
    
    plt.close()

# @input    srss_table: x,y, srss 컬럼을 포함한 데이터프레임
# @output   적절한 scale up factor 배열 반환
def get_scaleup_factor(table_srss, new=0):    
    # 응답 스펙트럼 데이터 불러오기
    df_response = get_response_spectrum()
    scale = []
    
    with open("config.json", "r") as json_file:
        config = json.load(json_file)
    T_min = config["T_min"]   

    # 각 지진파에 적당한 scale up factor 연산산
    for i in range(100):
        try:
            idx = str(i+1)
            X = table_srss[idx+'srss'].values.reshape(-1, 1)  # 입력 값 (df2)
            y = df_response['1.3*MCE*0.8'].fillna(0).values  # 출력 값 (df1)

            X_index = table_srss.index
            Y_index = df_response.index

            
            # 크기 조정 (X와 y 크기 맞추기)
            index_len = min(len(X), len(y))
            X = X[:index_len]
            y = y[:index_len]
            X_index = X_index[:index_len]
            Y_index = Y_index[:index_len]

            closest_indices = np.array([np.abs(Y_index - x).argmin() for x in X_index])
            matched_Y = y[closest_indices]

            # 배열을 numpy 배열로 변환
            recorded_spectrum = np.array(X)
            target_spectrum = np.array(matched_Y)
            periods = np.array(X_index)

            # 주기 간격 계산
            delta_T = np.diff(periods)
            delta_T = np.append(delta_T, delta_T[-1])  # 마지막 값은 이전 값과 동일하게 설정

            # 분자 계산
            numerator = np.sum(recorded_spectrum * target_spectrum * delta_T)

            # 분모 계산
            denominator = np.sum(recorded_spectrum**2 * delta_T)

            # 조정 계수 α 계산
            alpha = numerator / denominator

            scale.append(alpha)
        except KeyError:
            break


    # 결과 출력
    print(f"Best scale: {scale}")

    if new == 0:
        sf_filename = "./scale_up_factor.txt"
    else:
        sf_filename = "./new_scale_up_factor.txt"

    with open(sf_filename, 'w') as f:
        res = ""
        for item in scale:
            res = res + (str(float(item))+', ')
        f.write(res[:-2])
    return scale


# @input    srss_table: x,y, srss 컬럼을 포함한 데이터프레임
# @output   Step2/SRSS 위치에 srss table의 그래프 이미지 파일 저장장
def plot_srss(srss_table):
    plot_response(srss_table, PATH_STEP2+"/SRSS")

# @input    srss_table: x,y, srss 컬럼을 포함한 데이터프레임
#           scale: scale up factor를 저장하는 배열열
# @output   Step2/SRSS_scale 위치에 scaleup factor가 적용된 그래프의 이미지 파일을 저장장
def plot_scaleup_factor(table_srss, scale, path, mod='general'):

    if not os.path.exists(path):
        os.makedirs(path)

    # SRSS 그래프에 scale up factor 적용용
    table_srss_scale = table_srss.copy()
    for i in range(100):
        idx = str(i+1)
        try:
            table_srss_scale[idx+'srss'] = table_srss[idx+'srss'] * scale[i]
            table_srss_scale[idx+'x'] = table_srss[idx+'x'] * scale[i]
            table_srss_scale[idx+'y'] = table_srss[idx+'y'] * scale[i]
        except KeyError:
            break

    # Scale up factor 가 적용된 SRSS 그래프 생성
    plot_response(table_srss_scale, path, mod=mod)

    # Scale up factor 가 적용된 데이터를 Excel 파일에 저장하기
    output_file = path + '/srss_scale.xlsx'

    # Pandas의 ExcelWriter 사용
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1행에 Scale factor 정보 저장장
        df_numpy = pd.DataFrame([[''] * (3 * len(scale))])  # 빈 행 생성
        df_numpy[0] = "SF"
        for i, value in enumerate(scale):
            df_numpy.iloc[0, 3 * i + 1] = value  # 1, 4, 7... 열에 값 저장

        # sf 적용한 데이터 테이블 설정 및 순서 조정정
        table_srss_scale['T'] = table_srss.index

        table_srss_scale = table_srss_scale[["T"] + [f"{i}{suffix}" for i in range(1, int(idx)) for suffix in ["x", "y", "srss"]]]

        # 저장
        df_numpy.to_excel(writer, index=False, header=False, sheet_name='SRSS(Scale up)')
        table_srss_scale.to_excel(writer, startrow=1, startcol=0, index=False, sheet_name='SRSS(Scale up)')


def get_acceleration_table():
    dataset = pd.read_excel(FILE_NAME, sheet_name = 'acc')
    df = pd.DataFrame(dataset)

    return df

# @input    scale: scalue up factor들을 저장하는 배열열
# @output   Step2/Acceleration 위치에 scale up factor를 적용한 x,y방향 가속도도 그래프 이미지 파일 저장장
def plot_acceleration(df, scale, path,mod = 'general'):
    
    # 그래프 이미지 저장 폴더 생성
    if not os.path.exists(path):
        os.makedirs(path)

    # SRSS 그래프 그리기
    for i in range(100):
        idx = str(i+1)
                
        # x방향 그래프 그리기기
        plt.figure(figsize=(11, 6))
        try:
            
            plt.plot(df[idx+"xT"], df[idx+"xAcc"] * scale[i])
        except KeyError:
            break

        acc_abs = df[idx+"xAcc"].abs()
        acc_max_idx = acc_abs.idxmax()

        df[idx+"xAcc"] = df[idx+"xAcc"] * scale[i]
        acc_max = df.iloc[acc_max_idx][idx+"xAcc"]
        textbox_loc = 0.01
        if acc_max < 0:
            textbox_loc *= -1

        try:
            plt.plot(df.iloc[acc_max_idx][idx+"xT"], df.iloc[acc_max_idx][idx+"xAcc"], 'o', color='black')
            plt.text(df.iloc[acc_max_idx][idx+"xT"], df.iloc[acc_max_idx][idx+"xAcc"] + textbox_loc * 2, str(round(df.iloc[acc_max_idx][idx+"xAcc"],3))+"g", ha='center', va='top', fontsize=10)
            plt.axhline(y=0, color='black', linestyle='-')
        except IndexError:
            print(acc_max_idx, idx)
        # 그래프 꾸미기
        plt.title(idx+"X")         
        plt.ylim(acc_max * -120 * textbox_loc , acc_max * 120 * textbox_loc)  # 20% 여유
        plt.xlabel("Time(sec)")
        plt.ylabel("Acceleration(g)")

        # 그래프 출력
        plt.savefig(path + "/" + idx+"X.jpg")
        plt.clf()

        # x방향 엑셀파일 저장
        output_file = path + '/' + idx + 'x.csv'  # 저장할 CSV 파일 이름
        df_x = df[[idx+"xT",idx+"xAcc"]]  # 빈 행 생성
        df_x.to_csv(output_file, index=False, header=False) 
        
        if mod == 'text':
            # 헤더 문자열
            header_text = (
                f"Time Series matched accelerogram: {idx}X.txt\n"
                "Time Step: 0.02 s\n"
                "Time(sec)\tAcc(g)\n"
            )

            # TXT 파일로 저장 (탭 구분자로, 인덱스 없이)
            with open(path + '/' + idx + 'X.txt', 'w') as f:
                f.write(header_text)  # 헤더 먼저 쓰기
                df_x.to_csv(f, index=False, sep='\t', header=False)  # 데이터 쓰


        #y방향 그래프 그리기
        plt.figure(figsize=(11, 6))

        acc_abs = df[idx+"yAcc"].abs()
        acc_max_idx = acc_abs.idxmax()        

        df[idx+"yAcc"] = df[idx+"yAcc"] * scale[i]
        acc_max = df.iloc[acc_max_idx][idx+"yAcc"]
        textbox_loc = 0.01
        if acc_max < 0:
            textbox_loc *= -1

        
        plt.plot(df[idx+"yT"], df[idx+"yAcc"])
        plt.plot(df.iloc[acc_max_idx][idx+"yT"], df.iloc[acc_max_idx][idx+"yAcc"], 'o', color='black')
        plt.text(df.iloc[acc_max_idx][idx+"yT"], df.iloc[acc_max_idx][idx+"yAcc"] + textbox_loc * 2, str(round(df.iloc[acc_max_idx][idx+"yAcc"],3))+"g", ha='center', va='top', fontsize=10)
        plt.axhline(y=0, color='black', linestyle='-')

        # 그래프 꾸미기
        
        plt.title(idx+"Y")
        plt.ylim( acc_max  * -120 * textbox_loc, acc_max  * 120 * textbox_loc)  # 10% 여유
        plt.xlabel("Time(sec)")
        plt.ylabel("Acceleration(g)")

        # 그래프 출력
        plt.savefig(path + "/" + idx+"Y.jpg")
        plt.clf()

        # Pandas의 ExcelWriter 사용
        output_file = path + '/' + idx + 'y.csv'  # 저장할 CSV 파일 이름
        df_y = df[[idx + "yT", idx + "yAcc"]]  # 선택한 열 추출
        df_y.to_csv(output_file, index=False, header=False)  # CSV 파일로 저장

        if mod == 'text':
            # 헤더 문자열
            header_text = (
                f"Time Series matched accelerogram: {idx}Y.txt\n"
                "Time Step: 0.02 s\n"
                "Time(sec)\tAcc(g)\n"
            )

            # TXT 파일로 저장 (탭 구분자로, 인덱스 없이)
            with open(path + '/' + idx + 'Y.txt', 'w') as f:
                f.write(header_text)  # 헤더 먼저 쓰기
                df_y.to_csv(f, index=False, sep='\t', header=False)  # 데이터 쓰
    
    plt.close()

def Step2():
    cmd = 'N'
    while True:
        cmd = input("Step 2")
        if cmd == 'Y' or cmd =='y':
            break
    while True:
        cmd = input("waves.xlsx의 data를 입력해주세요 (y/N)")
        if cmd == 'Y' or cmd =='y':
            break
    srss_table = get_srss_table()
    scale = get_scaleup_factor(srss_table)

    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP2 + "/SRSS_scale")

    acc_table = get_acceleration_table()

    plot_acceleration(acc_table, scale, PATH_STEP2 + "/Acceleration")

    print()
def Step2_SF():    
    srss_table = get_srss_table()
    scale = get_scaleup_factor(srss_table)

    return srss_table, scale

def Step2_srss(srss_table, scale):
    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP2 + "/SRSS_scale", mod='total')

def Step2_acc(scale):
    acc_table = get_acceleration_table()
    plot_acceleration(acc_table, scale, PATH_STEP2 + "/Acceleration")

def main():
    srss_table = get_srss_table()
    scale = get_scaleup_factor(srss_table)

    plot_srss(srss_table)
    plot_scaleup_factor(srss_table, scale, PATH_STEP2 + "/SRSS_scale")

    acc_table = get_acceleration_table()

    plot_acceleration(acc_table, scale, PATH_STEP2 + "/Acceleration")
    
if __name__=="__main__":
    main()
# 2.4			0.6			0.9			0.9			0.25			3.5			1.5
