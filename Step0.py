import pandas as pd
from io import StringIO
import os
import re
import random
import json
import platform
import subprocess


# 가속도 데이터 폴더 경로 설정
folder_path = '.\\step0\\input'  # <- 여기 수정!

# SearchResults.csv 파일 경로
file_path = folder_path +'\\_SearchResults.csv'

# 특정 마커로 구간을 추출하는 함수
def extract_section(lines, start_marker, to_file_end=False):
    start_idx = None
    end_idx = None

    # 시작 인덱스 찾기
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    # 끝 인덱스 찾기
    if not to_file_end:
        for i in range(start_idx, len(lines)):
            if lines[i].strip() == '':
                end_idx = i
                break
    else:
        end_idx = len(lines)

    # 해당 구간 추출
    data_lines = lines[start_idx:end_idx]
    return data_lines

# 추출한 텍스트를 DataFrame으로 변환하는 함수
def lines_to_df(lines):
    if lines is None or len(lines) == 0:
        return None
    text = ''.join(lines)
    return pd.read_csv(StringIO(text))

# 가속도 파일 읽기 함수 
def read_acceleration_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # NPTS, DT 추출
    npts = None
    dt = None
    data_start_idx = None

    for idx, line in enumerate(lines):
        if "NPTS=" in line:
            npts_match = re.search(r'NPTS=\s*(\d+)', line)
            dt_match = re.search(r'DT=\s*([\d\.]+)', line)
            if npts_match and dt_match:
                npts = int(npts_match.group(1))
                dt = float(dt_match.group(1))
                data_start_idx = idx + 1
                break

    if npts is None or dt is None:
        raise ValueError(f"NPTS 또는 DT를 {file_path}에서 찾지 못했습니다.")

    # 가속도 데이터 읽기
    data_values = []
    for line in lines[data_start_idx:]:
        parts = line.strip().split()
        for p in parts:
            try:
                value = float(p)
                data_values.append(value)
            except ValueError:
                pass

    data_values = data_values[:npts]
    time_values = [i * dt for i in range(npts)]

    return time_values, data_values



# 결과를 저장할 배열
files = []
acc_all_serials_df = []
srss_all_serials_df=[]



def Step0():
    # AT2 파일을 순회
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # UP 제외 .AT2 파일만 수집
        at2_files = []
        for filename in filenames:
            if filename.lower().endswith('.at2') and 'UP' not in filename.upper():
                at2_files.append(os.path.join(dirpath, filename))
        
        if not at2_files:
            continue

        # serial별 그룹 만들기
        serial_groups = {}
        for filepath in at2_files:
            filename = os.path.basename(filepath)
            match = re.match(r'RSN(\d+)_', filename, re.IGNORECASE)
            if match:
                serial = match.group(1)
                if serial not in serial_groups:
                    serial_groups[serial] = []
                serial_groups[serial].append(filepath)

        # serial 그룹별로 처리
        for serial, paths in serial_groups.items():
            x_file = None
            y_file = None

            numbered_files = []
            ew_ns_files = []

            for path in paths:
                fname = os.path.basename(path)
                
                # 파일명 끝에서 숫자 추출
                number_match = re.search(r'(\d+)\.AT2$', fname, re.IGNORECASE)
                if number_match:
                    number = int(number_match.group(1))
                    numbered_files.append((number, path))
                else:
                    # E/W/N/S 판별
                    dir_match = re.search(r'([EWNS])[-.]?AT2$', fname, re.IGNORECASE)
                    if dir_match:
                        direction = dir_match.group(1).upper()
                        ew_ns_files.append((direction, path))

            # 우선 숫자 기준으로 분류
            if len(numbered_files) == 2:
                numbered_files.sort(key=lambda x: x[0])
                x_file = numbered_files[0][1]  # 작은 숫자
                y_file = numbered_files[1][1]  # 큰 숫자

            # E/W/N/S 기준 파일이 있으면 우선 적용
            for direction, path in ew_ns_files:
                if direction in ['E', 'W']:
                    x_file = path
                elif direction in ['N', 'S']:
                    y_file = path

            if x_file and y_file:
                files.append({
                    "serial": serial,
                    "x": x_file,
                    "y": y_file
                })
            else:
                print(f"[주의] RSN{serial} 그룹에서 X 또는 Y 파일을 찾지 못했습니다.")

    # 각각 섹션 추출
    # 파일을 한 줄씩 읽기
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        metadata_lines = extract_section(lines, "-- Summary of Metadata of Selected Records --")
        srss_y_lines = extract_section(lines, "-- Unscaled Horizontal & Vertical Spectra as recorded --", to_file_end=True)

    # 각각을 DataFrame으로 변환
    metadata = lines_to_df(metadata_lines)
    srss = lines_to_df(srss_y_lines)

    selected_files = random.sample(files,  min(7, len(files)))
    print(selected_files)

    df = pd.DataFrame({
        'T': srss['Period (sec)'],
    })
    srss_all_serials_df.append(df)

    for idx, item in enumerate(selected_files, 1):
        serial = item['serial']
        x_path = item['x']
        y_path = item['y']

        # 데이터프레임으로 변환
        df = pd.DataFrame({
            f'{idx}x': srss['RSN-'+serial+' Horizontal-1 pSa (g)'],
            f'{idx}y': srss['RSN-'+serial+' Horizontal-2 pSa (g)'],
        })
        srss_all_serials_df.append(df)

        # ACC 파일 읽기
        x_time, x_acc = read_acceleration_file(x_path)
        y_time, y_acc = read_acceleration_file(y_path)

        # 짧은 쪽에 맞추기
        min_len = min(len(x_time), len(y_time))
        x_time = x_time[:min_len]
        x_acc = x_acc[:min_len]
        y_time = y_time[:min_len]
        y_acc = y_acc[:min_len]

        # 데이터프레임으로 변환
        df = pd.DataFrame({
            f'{idx}xT': x_time,
            f'{idx}xAcc': x_acc,
            f'{idx}yT': y_time,
            f'{idx}yAcc': y_acc
        })

        acc_all_serials_df.append(df)

    # 가로로 붙이기
    acc_final_df = pd.concat(acc_all_serials_df, axis=1)

    srss_final_df = pd.concat(srss_all_serials_df, axis=1)


    # 파일 경로
    save_path = './waves.xlsx'

    # 3. pandas ExcelWriter를 사용해서 이어쓰기
    with pd.ExcelWriter(save_path, engine="openpyxl",mode='a', if_sheet_exists='replace') as writer:
        srss_final_df.to_excel(writer, sheet_name='data', index=False)
        acc_final_df.to_excel(writer, sheet_name='acc', index=False)

    
    # 1. 엑셀 읽기
    df = pd.read_excel(save_path, sheet_name='DBE', engine='openpyxl')

    # 2. 필요한 계산
    df['MCE'] = df['DBE'] * 1.5
    df['1.3*MCE*0.8'] = 1.3 * df['MCE'] * 0.8
    df['1.3*MCE*0.9'] = 1.3 * df['MCE'] * 0.9

    # 3. 엑셀에 덮어쓰기
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='DBE', index=False)

        # 1. 엑셀 읽기
    df = pd.read_excel(save_path, sheet_name='S1DBE', engine='openpyxl')

    # 2. 필요한 계산
    df['MCE(S1)'] = df['DBE(S1)'] * 1.5
    df['1.3*MCE*0.8(S1)'] = 1.3 * df['MCE(S1)'] * 0.8

    # 3. 엑셀에 덮어쓰기
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='S1DBE', index=False)



    # 저장할 경로
    files_save_path = './random_files.json'

    # 저장
    with open(files_save_path, 'w', encoding='utf-8') as f:
        json.dump(selected_files, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Execute when the module is not initialized from an import statement.
    Step0()