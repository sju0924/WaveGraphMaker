import pandas as pd
import matplotlib.pyplot as plt
from math import trunc
import json
import os

PATH_STEP1="./Step1"

# 그래프 x축 세부 범위 수정
def transform_x(x, T_min, ratio_tmin=0, ratio_second_half = {}):
    # 0~1까지 50에 매치
    # T1을 35에 매치
    transformed_x = 0

    if x <= T_min:
        transformed_x = ratio_tmin * x / T_min
    elif x > T_min and x < 1:
        transformed_x = ratio_tmin + ((50-ratio_tmin) * (x-T_min) / (1-T_min))    
    else:        
        x_int = trunc(x)        

        if x_int <= 10:
            transformed_x = ratio_second_half[x_int] + (ratio_second_half[x_int+1]-ratio_second_half[x_int])*(x-x_int)

        else:
            raise ValueError(f"x 값 {x}는 범위를 벗어났습니다.")

    return transformed_x

def get_response_spectrum(mode='DBE'):
    dataset = pd.read_excel('./waves.xlsx', sheet_name = mode)
    df = pd.DataFrame(dataset)

    return df

def plot_target_response_spectra(T_min, T_max, target):
    
    
    # 기존 DataFrame의 인덱스를 변환된 x축 값으로 설정
    with open("config.json", "r") as json_file:
        config = json.load(json_file)
    T_min = config["T_min"]
    T_max = config["T_max"]
    ratio_tmin = config["ratio_tmin"]
    ratio_second_half = config["ratio_second_half"] = {int(k): v for k, v in config["ratio_second_half"].items()}

    df = get_response_spectrum()
    df_transform = df.copy()
    df_transform.index = df_transform["T"].map(lambda x: transform_x(x, T_min*0.2, ratio_tmin, ratio_second_half))

    plt.figure(figsize=(11, 6))

    # x축: T, y축: 나머지 컬럼 값들
    for col in df_transform.columns[1:]:  # 첫 번째 컬럼(T)을 제외한 나머지
        plt.plot(df_transform.index, df_transform[col], label=col)

    # 건물 최대, 최소주기 표시
    tmin_transform = transform_x(T_min*0.2, T_min*0.2, ratio_tmin, ratio_second_half)
    plt.axvline(x=tmin_transform, color='red', linestyle='--')
    plt.text(
    tmin_transform, -0.05,
    f'0.2T({T_min * 0.2:.2f})',
    color='red', ha='center', va='top', fontsize=10
    )
    
    tmax_transform = transform_x(T_max*1.5, T_min*0.2, ratio_tmin, ratio_second_half)
    plt.axvline(x=tmax_transform, color='blue', linestyle='--')
    plt.text(
    tmax_transform, -0.05,
    f'1.5T({T_max * 1.5:.2f})',
    color='blue', ha='center', va='top', fontsize=10
    )

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

    # 그래프 꾸미기
    plt.title("Spectrum("+target+")")
    plt.xlabel("T")

    plt.xlim([0,100])
    plt.ylim([0,1])

    plt.legend()
    plt.grid(True)

    # 그래프 출력
    if not os.path.exists(PATH_STEP1):
        os.makedirs(PATH_STEP1)
    plt.savefig(PATH_STEP1 + "/Spectrum("+target+").png")
    plt.show()
    

def plot_s1_response_spectra(T_min, T_max):

    

     # 기존 DataFrame의 인덱스를 변환된 x축 값으로 설정
    with open("config.json", "r") as json_file:
        config = json.load(json_file)
    T_min = config["T_min"]
    T_max = config["T_max"]
    ratio_tmin = config["ratio_tmin"]
    ratio_second_half = config["ratio_second_half"] = {int(k): v for k, v in config["ratio_second_half"].items()}

    dfs1 = get_response_spectrum(mode='S1DBE')
    df_transform = dfs1.copy()
    df_transform.index = df_transform["T"].map(lambda x: transform_x(x, T_min*0.2, ratio_tmin, ratio_second_half))

    plt.figure(figsize=(11, 6))
    
    for col in df_transform.columns[1:]:  # 첫 번째 컬럼(T)을 제외한 나머지
        plt.plot(df_transform.index, df_transform[col], label=col)

    # 건물 최대, 최소주기 표시
    tmin_transform = transform_x(T_min*0.2, T_min*0.2, ratio_tmin, ratio_second_half)
    plt.axvline(x=tmin_transform, color='red', linestyle='--')
    plt.text(
    tmin_transform, -0.05,
    f'0.2T({T_min * 0.2:.2f})',
    color='red', ha='center', va='top', fontsize=10
    )

    tmax_transform = transform_x(T_max*1.5, T_min*0.2, ratio_tmin, ratio_second_half)
    plt.axvline(x=tmax_transform, color='blue', linestyle='--')
    plt.text(
    tmax_transform, -0.05,
    f'1.5T({T_max * 1.5:.2f})',
    color='blue', ha='center', va='top', fontsize=10
    )

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
    # 그래프 꾸미기
    plt.title("Spectrum(S1)")
    plt.xlabel("T(S1)")

    # S1 파동은 상수이니 가장 그래프가 이쁘게 나오게끔 범위 설정
    plt.xlim([0,100])
    plt.ylim([0,1])
    plt.legend()
    plt.grid(True)

    # 그래프 출력
    if not os.path.exists(PATH_STEP1):
        os.makedirs(PATH_STEP1)
    plt.savefig(PATH_STEP1 + "/Spectrum(S1).png")
    plt.show()

def Step1():
    # JSON 파일 읽기
    with open("config.json", "r") as json_file:
        config = json.load(json_file)

    # 읽은 데이터 사용
    T_min = config["T_min"]
    T_max = config["T_max"]
    target = 'S4'
    cmd = 'Y'

    while True:
        target = input("목표 응답 스펙트럼 종류를 입력해주세요 >> ")
        cmd = input("목표 응답 스펙트럼: " +target+" (Y/n)")
        if cmd == 'Y' or cmd =='y'or cmd == '':
            break

    plot_target_response_spectra(T_min, T_max, target)
    plot_s1_response_spectra(T_min, T_max)

    print("목표 응답스펙트럼 그래프 생성이 완료되었습니다.")

def main():
    # JSON 파일 읽기
    with open("config.json", "r") as json_file:
        config = json.load(json_file)

    # 읽은 데이터 사용
    T_min = config["T_min"]
    T_max = config["T_max"]

    target = 'S4'

    

    plot_target_response_spectra(T_min, T_max, target)
    plot_s1_response_spectra(T_min, T_max)

if __name__ == "__main__":
    # Execute when the module is not initialized from an import statement.
    main()
