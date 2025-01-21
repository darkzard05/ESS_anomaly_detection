import torch
from sklearn.metrics import f1_score, classification_report


# 테스트 단계
def testing(save_path, test_loader, batch_size, criterion, device):
    model = torch.load(f'./saved_model/{save_path}.pt') # 가장 높은 성능을 가진 모델 불러오기

    model.eval()  # 모델을 평가 모드로 설정
    batch_loss, total_t, correct_t = 0, 0, 0  # 배치 손실, 총 샘플 수, 올바른 예측 수 초기화
    preds, targets = [], []  # 예측 및 실제 레이블 저장 리스트 초기화

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for batch in test_loader:  # 테스트 데이터로더에서 배치 단위로 반복
            data_t = batch['data'].to(device).float().reshape(batch_size, 1, 1, -1)  # 입력 데이터 변형
            target_t = batch['label'].to(device).long()  # 정답 레이블

            outputs_t = model(data_t)  # 모델의 출력

            preds.append(outputs_t.argmax(dim=1))  # 예측 저장
            targets.append(target_t)  # 실제 레이블 저장

            batch_loss += criterion(outputs_t, target_t).item()  # 배치 손실 업데이트
            correct_t += (outputs_t.argmax(dim=1) == target_t).sum().item()  # 올바른 예측 수 증가
            total_t += target_t.size(0)  # 총 샘플 수 업데이트

        test_loss = batch_loss / len(test_loader)  # 평균 테스트 손실 계산
        test_acc = 100 * correct_t / total_t  # 테스트 정확도 계산

        preds = torch.cat(preds).cpu().numpy()  # 예측을 numpy 배열로 변환
        targets = torch.cat(targets).cpu().numpy()  # 실제 레이블을 numpy 배열로 변환

        report = classification_report(targets, preds, digits=4)  # 분류 보고서 생성
        f1score = f1_score(targets, preds, average='macro')

        # 테스트 성능을 출력하고 저장
        with open(f"./output/{save_path}_test.txt", "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")  # 테스트 손실 기록
            f.write(f"Test Accuracy: {test_acc:.4f}\n")  # 테스트 정확도 기록
            f.write(f"F1 Score: {f1score:.4f}\n")
            f.write("\nTest Classification Report:\n")  # 분류 보고서 헤더 기록
            f.write(report)  # 분류 보고서 기록

        # 테스트 손실과 정확도 출력
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"F1 Score: {f1score:.4f}")
        print("\nTest Classification Report:\n", report)  # 분류 보고서 출력