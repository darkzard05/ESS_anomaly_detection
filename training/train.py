import torch
from sklearn.metrics import f1_score, classification_report
from tqdm import trange


def training(save_path, model, num_epochs, train_loader, val_loader, batch_size, optimizer, criterion, device):
    # 변수 초기화
    train_loss, train_acc, val_loss, val_acc = [], [], [], []  # 손실과 정확도를 저장할 리스트
    best_f1 = 0  # 최고의 F1 점수를 저장할 변수
    epochs = trange(num_epochs, desc='training')  # 훈련 epoch를 위한 진행 바 생성

    for epoch in epochs:
        # 학습 단계
        model.train()  # 모델을 학습 모드로 설정
        running_loss, correct, total = 0, 0, 0  # 손실, 올바른 예측 수, 총 샘플 수 초기화

        for batch in train_loader:  # 훈련 데이터로더에서 배치 단위로 반복
            inputs = batch['data'].to(device).float().reshape(batch_size, 1, 1, -1)  # 입력 데이터 변형
            labels = batch['label'].to(device).long()  # 정답 레이블

            optimizer.zero_grad()  # 기울기 초기화

            outputs = model(inputs)  # 모델의 출력
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            running_loss += loss.item()  # 총 손실 업데이트
            correct += torch.sum(outputs.argmax(dim=1) == labels).sum().item()  # 올바른 예측 수 증가
            total += labels.size(0)  # 총 샘플 수 업데이트

        train_loss.append(running_loss / len(train_loader))  # 평균 훈련 손실 저장
        train_acc_epoch = 100 * correct / total  # 훈련 정확도 계산
        train_acc.append(train_acc_epoch)  # 훈련 정확도 저장

        # 검증 단계
        model.eval()  # 모델을 평가 모드로 설정
        val_running_loss, total_t, correct_t = 0, 0, 0  # 검증 손실, 올바른 예측 수, 총 샘플 수 초기화
        preds, targets = [], []  # 예측 및 실제 레이블 저장 리스트 초기화

        with torch.no_grad():  # 그래디언트 계산 비활성화
            for batch in val_loader:  # 검증 데이터로더에서 배치 단위로 반복
                data_t = batch['data'].to(device).float().reshape(batch_size, 1, 1, -1)  # 입력 데이터 변형
                target_t = batch['label'].to(device).long()  # 정답 레이블
                
                outputs_t = model(data_t)  # 모델의 출력

                preds.append(outputs_t.argmax(dim=1))  # 예측 저장
                targets.append(target_t)  # 실제 레이블 저장

                val_running_loss += criterion(outputs_t, target_t).item()  # 검증 손실 업데이트
                correct_t += (outputs_t.argmax(dim=1) == target_t).sum().item()  # 올바른 예측 수 증가
                total_t += target_t.size(0)  # 총 샘플 수 업데이트

            val_loss_epoch = val_running_loss / len(val_loader)  # 평균 검증 손실 계산
            val_acc_epoch = 100 * correct_t / total_t  # 검증 정확도 계산

            val_loss.append(val_loss_epoch)  # 검증 손실 저장
            val_acc.append(val_acc_epoch)  # 검증 정확도 저장

            preds_ = torch.cat(preds).cpu().numpy()  # 예측을 numpy 배열로 변환
            targets_ = torch.cat(targets).cpu().numpy()  # 실제 레이블을 numpy 배열로 변환

            f1score = f1_score(targets_, preds_, average='macro')  # F1 점수 계산
            if f1score > best_f1:  # 현재 F1 점수가 최상의 경우
                best_f1 = f1score  # 최고 F1 점수 업데이트
                torch.save(model, f'./saved_model/{save_path}.pt')  # 모델 저장
                with open(f"./output/{save_path}.txt", "a") as f:
                    f.write(f'Epoch: {epoch}\n')  # 에폭 정보 기록
                    f.write(classification_report(targets_, preds_, digits=4))  # 분류 보고서 기록
                print('model save')  # 모델 저장 메시지 출력

            # 훈련 및 검증 정확도 출력
            print(f"Epoch {epoch+1} - Train Acc: {train_acc_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, F1 Score: {f1score:.4f}, Best F1: {best_f1:.4f}")
            epochs.set_postfix_str(f"Train Acc = {train_acc_epoch:.4f}, Val Acc = {val_acc_epoch:.4f}, F1 = {f1score:.4f}, Best F1 = {best_f1:.4f}")