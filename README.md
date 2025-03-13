# 🧠 뇌종양 진단 프로젝트

## 📌 프로젝트 개요
이 프로젝트는 **뇌종양 MRI 데이터**를 활용하여 **Convolutional Neural Network(CNN)** 모델을 개발하고, 다양한 모델을 비교 분석하여 최적의 성능을 가진 모델을 선정하는 것을 목표로 합니다.

- **데이터셋**: Kaggle에서 제공하는 뇌종양 MRI 데이터셋
- **프레임워크**: PyTorch
- **모델 비교**: `BasicCNN`, `IntermediateCNN`, `AdvancedCNN`
- **평가 지표**: Accuracy, Precision, Recall, F1-score

## 🔬 데이터 전처리
- **이미지 크기 조정**: `224x224`
- **정규화**: `PyTorch Tensor` 형식 변환 후 평균 및 표준편차 정규화 적용
- **데이터 분할**: `Train`, `Validation`, `Test` (배치 크기: 128)

## 🏗️ 모델 구조
### 1️⃣ BasicCNN
- 2개의 **합성곱 레이어** (Convolutional Layers)
- 1개의 **완전 연결층** (Fully Connected Layer)
- 연산량이 적고 빠르게 학습 가능하지만 성능이 낮음

### 2️⃣ IntermediateCNN
- 4개의 **합성곱 레이어**
- **드롭아웃(Dropout)** 적용하여 과적합 방지
- BasicCNN보다 성능이 향상됨

### 3️⃣ AdvancedCNN
- 6개의 **합성곱 레이어**
- **배치 정규화(Batch Normalization)** 및 **드롭아웃** 활용하여 학습 안정성 증가
- 가장 높은 성능을 기록함

## ⚙️ 학습 설정
- **손실 함수**: Negative Log Likelihood Loss (NLLLoss)
- **최적화 알고리즘**: Adam Optimizer
- **에포크**: 30 (Early Stopping 적용)
- **모델 가중치 저장**: 최적의 성능을 가진 모델을 저장하여 활용

## 📊 모델 성능 비교
| Model           | Accuracy |
|----------------|----------|
| BasicCNN       | 89.23%   |
| IntermediateCNN| 95.10%   |
| AdvancedCNN    | 96.01%   |

AdvancedCNN 모델이 가장 높은 정확도를 기록하여 최적의 모델로 선정되었습니다.

## 🔮 향후 연구 방향
- **데이터 증강(Data Augmentation)** 기법 적용하여 일반화 성능 향상
- **하이퍼파라미터 튜닝(Hyperparameter Tuning)** 활용 최적의 학습률과 배치 크기 탐색
- **사전 훈련된 모델(Pretrained Models)** 활용 (e.g., `ResNet`, `EfficientNet`)
- 실험적인 최적화 기법을 적용하여 더욱 정교한 모델 개발

## 🚀 실행 방법

데이터셋이 필요하다면, [여기](https://github.com/MLMedical9707/BRAIN_TUMOR_SEGMENTATION/blob/main/BRAIN_TUMOR_SEGMENTATION.ipynb)에서 다운로드할 수 있습니다.

## 📄 라이선스
이 프로젝트는 **MIT License** 하에 배포됩니다.
