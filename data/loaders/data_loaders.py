import torch
from torch.utils.data import Dataset, DataLoader

# 사용자 정의 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data):
        # 데이터프레임에서 특성과 레이블을 분리
        self.data = data.iloc[:, :-1].values  # 모든 행과 마지막 열을 제외한 데이터
        self.labels = data.iloc[:, -1].values.astype(int)  # 마지막 열을 레이블로 사용하고 정수형으로 변환
        
    def __len__(self):
        # 데이터셋의 크기를 반환
        return len(self.data)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 데이터와 레이블을 반환
        sample = {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),  # 데이터를 텐서로 변환
            'label': torch.tensor(self.labels[idx], dtype=torch.long)   # 레이블을 텐서로 변환
        }
        return sample