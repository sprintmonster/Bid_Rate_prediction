# Quantile Regression + KDE 기반 확률 회귀 모델

이 문서는 Quantile Regression과 Kernel Density Estimation(KDE)을 결합한 비모수적 확률 예측 모델의 구조와 학습 방식을 설명합니다.

## 📋 모델 개요

### 핵심 아이디어

- **Quantile Regression**: 타겟 변수의 조건부 분위수(0.1, 0.5, 0.9 등)를 직접 학습
- **KDE 변환**: 학습된 quantile들을 KDE로 스무딩 → 연속적 **확률밀도함수(PDF)** 생성

![모델 아키텍처 개요][file:20]

## 🎯 1. Pinball Loss (분위수 손실)

분위수 τ에 대한 asymmetric loss로, 예측 오차 방향별 가중치 차등 적용.

**수식**:


![Pinball Loss 시각화][file:16]

**특징**:
- **τ > 0.5**: 과소추정 시 더 큰 패널티 (상승 슬로프 τ)
- **τ < 0.5**: 과대추정 시 더 큰 패널티 (하강 슬로프 τ-1)

![Pinball Loss 상세][file:17]

## 🔗 2. Non-Crossing 제약

quantile 레벨 τ_i < τ_j 일 때 `ŷ_τi ≤ ŷ_τj` 보장.

**Non-crossing Loss**:


**최종 Objective**:

![Non-crossing 제약][file:18]

## 🔄 3. KDE 변환 과정

학습된 quantile 값들을 KDE로 PDF 재구성.

**Transformer KDE**:

![KDE 변환 과정][file:21]

## 🏗️ 4. 모델 아키텍처

![전체 모델 구조][file:20]

**구성**:
- **Transformer Encoder**: window_size × features 입력 처리
- **Quantile Heads**: 병렬 MLP (hidden dim 사용자 정의)
- **KDE Post-processing**: quantile → PDF 변환

## ⚖️ 5. 종합 Objective Function

![종합 손실함수][file:19]

## 📊 6. 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **Coverage** | 실제값이 PI 안에 포함되는 비율 | nominal coverage (90% PI → 90%) |
| **Sharpness** | 예측구간 폭 | 최소화 |

## 💻 구현 가이드

```python
class QuantileKDEModel(nn.Module):
    def __init__(self, quantiles=[0.1,0.5,0.9], window_size=32):
        self.encoder = TransformerEncoder(window_size)
        self.heads = nn.ModuleList([MLP() for _ in quantiles])
    
    def forward(self, x):
        h = self.encoder(x)
        quantiles = torch.stack([head(h) for head in self.heads], dim=-1)
        pdf = kde_from_quantiles(quantiles)
        return pdf, quantiles

![종합 손실함수][file:19]

## 📊 6. 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **Coverage** | 실제값이 PI 안에 포함되는 비율 | nominal coverage (90% PI → 90%) |
| **Sharpness** | 예측구간 폭 | 최소화 |

## 💻 구현 가이드

```python
class QuantileKDEModel(nn.Module):
    def __init__(self, quantiles=[0.1,0.5,0.9], window_size=32):
        self.encoder = TransformerEncoder(window_size)
        self.heads = nn.ModuleList([MLP() for _ in quantiles])
    
    def forward(self, x):
        h = self.encoder(x)
        quantiles = torch.stack([head(h) for head in self.heads], dim=-1)
        pdf = kde_from_quantiles(quantiles)
        return pdf, quantiles
quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
KDE bandwidth: 0.1~1.0
λ_nc: 0.01~0.1


