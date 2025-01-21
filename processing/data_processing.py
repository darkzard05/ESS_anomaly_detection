import os
import pandas as pd


# 주어진 디렉토리에서 특정 접미어와 CSV 파일을 찾는 함수
def find_csv_with_suffix(directory, suffix):
    try:
         # 디렉토리 내의 파일 및 폴더 목록을 가져옴
        items = os.listdir(directory)
        for item in items:
             # 파일인지 확인하고, CSV 파일인지 및 특정 접미어로 끝나는지 검사
            if os.path.isfile(os.path.join(directory, item)) and item.endswith('.csv') and item.endswith(suffix):
                # 조건에 맞는 CSV 파일 경로를 반환
                return os.path.join(directory, item)
        # 조건에 맞는 파일이 없을 경우 None 반환
        return None
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.") # 디렉토리가 존재하지 않을 경우 에러 메시지 출력
        return None
    except PermissionError:
        print(f"Error: Permission denied to access the directory '{directory}'.") # 권한이 없을 경우 에러 메시지 출력
        return None
    
# CSV 파일을 찾고 읽어오는 함수
def process_csv_files(sub_path, condition, skiprows, result_list):
    # 조건에 맞는 CSV 파일을 찾기 위한 경로 생성
    target_path = os.path.join(sub_path, condition, 'Spreadsheets')
    try:
        # 해당 디렉토리 내에 조건에 맞는 CSV 파일이 있는지 확인하고 읽어오기
        csv_file = find_csv_with_suffix(target_path, 'Waveform.csv')
        if csv_file:
            print(f"Found CSV file: {csv_file}") # 찾은 CSV 파일 경로 출력
            df = pd.read_csv(csv_file, skiprows=skiprows) # CSV 파일을 읽어서 데이터프레임으로 변환
            result_list.append(df) # 읽어온 데이터프레임을 결과 리스트에 추가
        else:
            print(f"No CSV file ending with Waveform found in {target_path}") # 조건에 맞는 파일이 없을 경우 메시지 출력
    except FileNotFoundError:
        print(f"Directory {target_path} does not exist.") # 디렉토리가 존재하지 않을 경우 메시지 출력


def process_all_files_in_paths(paths, normal_list, sag_list, rvc_list, wave_change_list):
    # 각 날짜별로 경로 탐색 및 파일 처리
    for path in paths:
        for i in range(1, 32): # 1일부터 31일까지 반복
            sub_path = os.path.join(path, f'Day_{i:02d}') # 날짜별 서브 경로 생성
            
            # 존재하지 않는 디렉토리는 패스
            if not os.path.exists(sub_path):
                continue   # 서브 경로가 존재하지 않으면 다음 날짜로 넘어감

            # 파일 및 디렉토리 탐색
            items = os.listdir(sub_path) # 서브 경로 내의 모든 파일 및 디렉토리 목록 가져오기
            for item in items:
                # 조건에 따른 처리
                suffix = item.split('_')[-1]      # 파일명에서 접미사 추출
                pre_suffix = item.split('_')[-2]    # 파일명에서 이전 접미사 추출

                # 조건에 따라 적절한 CSV 파일 처리 함수 호출
                if suffix == 'Snapshot':
                    # 'Snapshot' 파일인 경우
                    process_csv_files(sub_path, item, 18, normal_list)
                elif suffix == 'Sag':
                    # 'Sag' 파일인 경우
                    if pre_suffix == 'Voltage':
                        # 이전 접미사가 'Voltage'인 경우
                        process_csv_files(sub_path, item, 22, sag_list)
                    else:
                        # 이전 접미사가 'Voltage'가 아닌 경우
                        process_csv_files(sub_path, item, 21, sag_list)
                elif suffix == 'RVC':
                    # 'RVC' 파일인 경우
                    process_csv_files(sub_path, item, 22, rvc_list)
                elif suffix == 'Change':
                    # 'Wave Change' 파일인 경우
                    process_csv_files(sub_path, item, 19, wave_change_list)
    return normal_list, sag_list, rvc_list, wave_change_list


def split_dataframes(split, length, normal_list, sag_list, rvc_list, wave_change_list):
     # 각 상태에 대한 분할된 데이터프레임 리스트 초기화
    df_normal_split = []  # 정상 상태 데이터프레임 분할 리스트
    df_sag_split = []  # Sag 상태 데이터프레임 분할 리스트
    df_rvc_split = []  # RVC 상태 데이터프레임 분할 리스트
    df_wave_change_split = []  # Wave Change 상태 데이터프레임 분할 리스트

    # 정상 상태 데이터프레임을 지정된 길이로 분할
    for df in normal_list:
        for i in range(0, length, length // split):  # 분할 크기만큼 반복
            df_normal_split.append(df.iloc[i:i + length // split, 1:-1])  # 각 조각을 리스트에 추가

    # Sag 상태 데이터프레임을 지정된 길이로 분할
    for df in sag_list:
        for i in range(0, length, length // split):
            df_sag_split.append(df.iloc[i:i + length // split, 1:-1])  # 각 조각을 리스트에 추가

    # RVC 상태 데이터프레임을 지정된 길이로 분할
    for df in rvc_list:
        for i in range(0, length, length // split):
            df_rvc_split.append(df.iloc[i:i + length // split, 1:-1])  # 각 조각을 리스트에 추가

    # Wave Change 상태 데이터프레임을 지정된 길이로 분할
    for df in wave_change_list:
        for i in range(0, length, length // split):
            df_wave_change_split.append(df.iloc[i:i + length // split, 1:-1])  # 각 조각을 리스트에 추가
    
    return df_normal_split, df_sag_split, df_rvc_split, df_wave_change_split

def flatten_and_concat(df_normal_split, df_sag_split, df_rvc_split, df_wave_change_split):
    # 각 상태에 대한 평탄화된 데이터프레임 리스트 초기화
    flatten_normal = []
    flatten_sag = []
    flatten_rvc = []
    flatten_wave_change = []

    # 정상 상태 데이터프레임 평탄화
    for df in df_normal_split:
        # 각 열을 연결하여 하나의 시리즈로 평탄화
        flatten_df = pd.concat([df[col] for col in df.columns], ignore_index=True)
        flatten_normal.append(flatten_df)  # 평탄화된 시리즈를 리스트에 추가

    # SAG 상태 데이터프레임 평탄화
    for df in df_sag_split:
        flatten_df = pd.concat([df[col] for col in df.columns], ignore_index=True)
        flatten_sag.append(flatten_df)  # 평탄화된 시리즈를 리스트에 추가

    # RVC 상태 데이터프레임 평탄화
    for df in df_rvc_split:
        flatten_df = pd.concat([df[col] for col in df.columns], ignore_index=True)
        flatten_rvc.append(flatten_df)  # 평탄화된 시리즈를 리스트에 추가

    # 파형 변화 상태 데이터프레임 평탄화
    for df in df_wave_change_split:
        flatten_df = pd.concat([df[col] for col in df.columns], ignore_index=True)
        flatten_wave_change.append(flatten_df)  # 평탄화된 시리즈를 리스트에 추가

    # 평탄화된 데이터를 열 방향으로 연결하고 전치하여 최종 데이터프레임 생성
    df_normal_total = pd.concat(flatten_normal, axis=1).transpose()
    df_sag_total = pd.concat(flatten_sag, axis=1).transpose()
    df_rvc_total = pd.concat(flatten_rvc, axis=1).transpose()
    df_wave_change_total = pd.concat(flatten_wave_change, axis=1).transpose()

    return df_normal_total, df_sag_total, df_rvc_total, df_wave_change_total