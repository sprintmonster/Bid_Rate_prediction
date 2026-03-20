## 🏗️ 상세 구현 내용 (AI/ML)

### 📊 **사정율 예측 AI 모델** ⭐

<div align="center">
  <!-- 문제 정의 -->
  <img src="./ops/images/39.png" alt="문제 정의" width="700"/>
  <br><b>Quantile Regression (10개 Quantile) + KDE</b>
</div>

#### 1️⃣ **모델 아키텍쳐**
<div align="center">
  <!-- 모델 아키텍쳐 -->
  <img src="./ops/images/40.png" alt="모델 아키텍쳐" width="500"/>
  <br><sub>L_τ(ŷ-y) = max(τ(ŷ-y), (τ-1)(ŷ-y))</sub>
</div>

#### 3️⃣ **Objective Function**
<div align="center">
  <!-- Objective Function -->
  <img src="./ops/images/41.png" alt="Objective Function" width="600"/>
  <br><sub>f̂_KDE(y) = 1/(11h) Σ K((y - q̂_τi)/h)</sub>
</div>

#### 2️⃣ **Non-Crossing 제약**
<div align="center">
  <!-- Non-crossing 데모 이미지 -->
  <img src="./ops/images/42.png" alt="Non-crossing 제약" width="500"/>
  <br><sub>L_nc = Σ_(i<j) max(0, q̂_τi - q̂_τj)</sub>
</div>

#### 3️⃣ **KDE 변환**
<div align="center">
  <!-- KDE 변환 과정 이미지 -->
  <img src="./ops/images/44.png" alt="Quantile → KDE 변환" width="600"/>
  <br><sub>f̂_KDE(y) = 1/(11h) Σ K((y - q̂_τi)/h)</sub>
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
