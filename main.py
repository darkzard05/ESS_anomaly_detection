import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

from processing.data_processing import process_all_files_in_paths, split_dataframes, flatten_and_concat
from data.loaders.data_loaders import CustomDataset
from training.train import training
from training.test import testing
from training.utils import create_models


def main(path, batch_size, num_epochs, split, length):
    # 경로 리스트와 결과 리스트 정의
    paths = [path + '/2024/Month_07/', path + '/2024/Month_08/', path + '/2024/Month_09/',
            path + '/2024/Month_10/', path + '/2024/Month_11/', path + '/2024/Month_12/']

    normal_list, sag_list, rvc_list, wave_change_list = process_all_files_in_paths(paths, normal_list, sag_list, rvc_list, wave_change_list)

    # 각 상태 데이터프레임을 고정된 길이로 분할
    df_normal_split, df_sag_split, df_rvc_split, df_wave_change_split = split_dataframes(split, length, normal_list, sag_list, rvc_list, wave_change_list)

    # 평탄화된 데이터를 열 방향으로 연결하고 전치하여 최종 데이터프레임 생성
    df_normal_total, df_sag_total, df_rvc_total, df_wave_change_total = flatten_and_concat(df_normal_split, df_sag_split, df_rvc_split, df_wave_change_split)

    # 라벨 추가
    df_normal_total['label'] = 0
    df_sag_total['label'] = 1
    df_rvc_total['label'] = 2
    df_wave_change_total['label'] = 0

    # 데이터 병합
    df_total = pd.concat([df_normal_total, df_sag_total, df_rvc_total, df_wave_change_total], axis=0)
    df_total.reset_index(drop=True, inplace=True)

    # 전체 데이터프레임을 60%의 훈련 데이터와 40%의 테스트 데이터로 분할
    train, test = train_test_split(df_total, test_size=0.4, random_state=77, shuffle=True, stratify=df_total['label'])

    # 테스트 데이터를 다시 50%로 나누어 20%의 검증 데이터와 20%의 최종 테스트 데이터로 분할
    val, test = train_test_split(test, test_size=0.5, random_state=77, shuffle=True, stratify=test['label'])

    # 인덱스를 재설정하여 연속된 정수형 인덱스 부여 (drop=True로 이전 인덱스 열 제거)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # 각 데이터셋에서 레이블의 분포를 출력
    print(train['label'].value_counts())
    print(val['label'].value_counts())
    print(test['label'].value_counts())

    # MinMaxScaler 객체 생성: 모든 특성을 0과 1 사이로 스케일링
    scaler = MinMaxScaler()

    # 훈련 데이터에 스케일러를 적용
    train_scaled = scaler.fit_transform(train.iloc[:, :-1])  # 레이블 제외한 모든 열에 적용

    # 검증 및 테스트 데이터에 기존에 적합된 스케일러를 사용하여 변환
    val_scaled = scaler.transform(val.iloc[:, :-1])  # 레이블 제외한 모든 열에 적용
    test_scaled = scaler.transform(test.iloc[:, :-1])  # 레이블 제외한 모든 열에 적용

    # 스케일된 훈련, 검증, 테스트 데이터와 레이블을 결합하여 새로운 데이터프레임 생성
    train = pd.concat([pd.DataFrame(train_scaled), train.iloc[:, -1]], axis=1)  # 훈련 데이터프레임
    val = pd.concat([pd.DataFrame(val_scaled), val.iloc[:, -1]], axis=1)  # 검증 데이터프레임
    test = pd.concat([pd.DataFrame(test_scaled), test.iloc[:, -1]], axis=1)  # 테스트 데이터프레임

    # 훈련 데이터의 마지막 열(레이블)의 고유한 값과 각 값의 개수를 계산
    unique_values, counts = np.unique(train.iloc[:, -1], return_counts=True)
    print(unique_values, counts)

    # SMOTE 기법을 사용하여 데이터셋의 불균형 문제를 해결하기 위한 객체 생성
    # k_neighbors는 생성할 이웃 샘플의 수를 지정
    smote = SMOTE(k_neighbors=5, random_state=42)

    # 훈련 데이터(X_train)와 레이블(y_train)에 대해 SMOTE를 적용하여 오버샘플링 수행
    X_train_resampled, y_train_resampled = smote.fit_resample(train.iloc[:, :-1], train.iloc[:, -1])

    # 리샘플링된 훈련 데이터에서 고유한 클래스 레이블과 해당 클래스의 개수를 계산
    unique_values, counts = np.unique(y_train_resampled, return_counts=True)
    print(unique_values, counts)

    # 리샘플링된 훈련 데이터와 레이블을 결합하여 새로운 데이터프레임 생성
    train = pd.concat(
        [pd.DataFrame(X_train_resampled), pd.DataFrame(y_train_resampled, columns=['label'])],
        axis=1  # 열 방향으로 결합
    )

    # 데이터프레임에서 각 레이블의 개수를 계산하고 출력
    print(train['label'].value_counts())
    print(val['label'].value_counts())
    print(test['label'].value_counts())

    # 커스텀 데이터셋 인스턴스 생성
    # 데이터셋을 위한 데이터로더 생성
    train_dataset = CustomDataset(train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = CustomDataset(val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = CustomDataset(test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # GPU 사용 가능 여부 확인
    USE_CUDA = torch.cuda.is_available()  # GPU를 사용 가능하면 True, 아니면 False를 리턴

    # 사용 가능한 장치 설정 (GPU 또는 CPU)
    device = torch.device("cuda" if USE_CUDA else "cpu")  # GPU 사용 가능하면 사용하고, 그렇지 않으면 CPU 사용
    print("다음 기기로 학습합니다:", device)  # 사용될 장치 출력

    # 모델 생성
    model = create_models().to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 크로스 엔트로피 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=2e-4)  # Adam 옵티마이저 설정

    # 결과 저장을 위한 디렉토리 생성
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./saved_model', exist_ok=True)

    # 학습 진행
    training(save_path, model, num_epochs, train_loader, val_loader, batch_size, optimizer, criterion, device)

    # 테스트 진행
    testing(save_path, test_loader, batch_size, criterion, device)    

if __name__ == '__main__':
    batch_size = 64
    num_epochs = 1
    
    split = 64  # 분할할 조각의 개수
    length = 4096  # 원본 데이터의 전체 길이

    path = './data/raw/gimhae_ESS'
    save_path = '1224_ESS_anomaly_detection_resnet18_csa_SMOTE'
    main(path, batch_size, num_epochs, split, length)