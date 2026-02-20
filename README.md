## 🏗️ 상세 구현 내용 (AI/ML)

### 📊 **사정율 예측 AI 모델** ⭐

<div align="center">
  <!-- 전체 모델 아키텍처 이미지 -->
  <img src="./ops/images/ai_model_architecture.png" alt="AI 모델 전체 아키텍처" width="700"/>
  <br><b>Quantile Regression (11개 Quantile) + KDE</b>
</div>

#### 1️⃣ **Pinball Loss**
<div align="center">
  <!-- Pinball Loss 그래프 -->
  <img src="./ops/images/pinball_loss.png" alt="Pinball Loss 함수" width="500"/>
  <br><sub>L_τ(ŷ-y) = max(τ(ŷ-y), (τ-1)(ŷ-y))</sub>
</div>

#### 2️⃣ **Non-Crossing 제약**
<div align="center">
  <!-- Non-crossing 데모 이미지 -->
  <img src="./ops/images/non_crossing_constraint.png" alt="Non-crossing 제약" width="500"/>
  <br><sub>L_nc = Σ_(i<j) max(0, q̂_τi - q̂_τj)</sub>
</div>

#### 3️⃣ **KDE 변환**
<div align="center">
  <!-- KDE 변환 과정 이미지 -->
  <img src="./ops/images/kde_transformation.png" alt="Quantile → KDE 변환" width="600"/>
  <br><sub>f̂_KDE(y) = 1/(11h) Σ K((y - q̂_τi)/h)</sub>
</div>

#### 4️⃣ **최종 Objective**
<div align="center">
  <!-- 종합 손실 함수 이미지 -->
  <img src="./ops/images/total_objective.png" alt="최종 손실 함수" width="500"/>
  <br><sub>L_total = 1/11 Σ L_τ + λ L_nc</sub>
</div>

---

### 🔢 **데이터 분석**
<div align="center">
  <!-- 데이터 분포/특성 이미지 -->
  <img src="./ops/images/data_analysis.png" alt="나라장터 데이터 분석" width="700"/>
  <br><sub>588,109건 (2021~2025) 전처리 완료</sub>
</div>

### 🧠 **RAG + 멀티모달 Agent**
<div align="center">
  <!-- RAG 파이프라인 이미지 -->
  <img src="./ops/images/rag_pipeline.png" alt="RAG + Agent 시스템" width="700"/>
</div>

---

**📁 이미지 파일 경로**: `./ops/images/`
