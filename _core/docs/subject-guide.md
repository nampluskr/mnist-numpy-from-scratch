---
tags: [templates, docs]
created: 2026-06-15
updated: 2026-06-15
---

# subject-guide.md

이 워크스페이스에서 공통으로 사용하는 주제 분류 기준을 정의한다.
문서 작성, 파일명, 태그, frontmatter 키워드를 결정할 때 이 분류를 기준으로 삼는다.

## 1. 개요

전체 분류는 연구, 학습, 강의자료, 개발환경 문서화를 포괄하도록 구성한다.
각 subject code는 문서 frontmatter 태그, 파일명 접두사, 색인, 검색 키워드에 활용한다.

Subject code 사용 방식은 다음과 같다.

- 문서 메타정보 태그 예시: `tags: [ML, classification, pytorch]`
- 폴더명 접두사 예시: `ml-classification/`, `nlp-tokenization/`
- 파일명 접두사 예시: `ml_image_classifier.py`, `nlp_tokenizer.md`

## 2. 주제 분류 목록

현재 정의된 주제 분류 코드와 Subject 명칭은 다음과 같다.

| No | Subject Code | Subject Name |
|---:|---|---|
| 01 | `MATH` | Mathematics |
| 02 | `PHYS` | Physics |
| 03 | `NA` | Numerical Analysis |
| 04 | `ML` | Machine Learning |
| 05 | `CV` | Computer Vision |
| 06 | `NLP` | Natural Language Processing |
| 07 | `CS` | Computer Science |
| 08 | `DEV` | Development Environment and Tools |
| 09 | `MISC` | Miscellaneous |

## 3. 주제별 세부 내용

각 분류 코드에 해당하는 주제가 다루는 세부 내용을 정리한다.

### 3.1. [MATH] Mathematics

해석학(Analysis)은 실수 체계에서 수열과 급수의 수렴, 연속성, 미분가능성, Riemann 적분을 엄밀하게 다룬다. Real Analysis는 함수의 균등수렴과 거리공간의 위상적 성질을 포함하며, Complex Analysis로 확장되면 residue theorem과 conformal mapping을 통한 적분 계산 도구를 제공한다.

대수학 분야에서는 Linear Algebra를 중심으로 벡터공간, 선형사상, 고유값 분해, 행렬 분해(QR / SVD / Cholesky)를 다룬다. Vector Calculus에서는 gradient·divergence·curl과 Stokes / Gauss 정리를 다루며, 미분방정식은 ODE와 PDE를 모두 포함한다. Special Functions 분야에서는 Gamma 함수, Bessel 함수, Legendre 다항식의 성질과 점화식을 다룬다.

세부 항목은 다음과 같다.

- Real Analysis
- Complex Analysis
- Vector Calculus
- Linear Algebra
- Differential Equations
- Mathematical Physics
- Special Functions
- Riemann Zeta Function

### 3.2. [PHYS] Physics

전자기학(Electromagnetism)은 Electrostatics와 Magnetostatics를 출발점으로 시간 변화하는 전자기장의 Electrodynamics까지 다룬다. Maxwell 방정식을 중심으로 전자기파의 발생과 전파를 이해하며, 수치 시뮬레이션과 직접 연결된다.

양자역학(Quantum Mechanics)은 파동함수, Schrödinger 방정식, 연산자 이론을 중심으로 미시 세계의 물리 법칙을 기술한다. 유체역학(Fluid Dynamics)과 플라즈마 물리(Plasma Physics)는 Navier-Stokes 방정식 기반 유체 흐름과 전자기장·하전 입자의 상호작용을 다루며, Plasma Simulation은 FDM / PIC 등 수치 기법을 활용한다.

세부 항목은 다음과 같다.

- Electromagnetism
- Electrostatics
- Magnetostatics
- Electrodynamics
- Quantum Mechanics
- Fluid Dynamics
- Plasma Physics
- Plasma Simulation

### 3.3. [NA] Numerical Analysis

수치해석은 해석적 방법으로 풀기 어려운 수학 문제를 컴퓨터 계산으로 근사하는 방법론을 다룬다. 비선형 방정식의 근 탐색(Newton-Raphson, bisection), 선형계(LU 분해, 반복법), 보간과 근사(Lagrange, spline)를 기초로 삼는다.

수치 미분·적분은 Gaussian quadrature, Runge-Kutta를 포함한 ODE 수치 풀이, FDM / FEM / FVM 등 PDE 수치 풀이까지 포괄한다. 최적화(Optimization) 분야에서는 gradient descent 계열, 선형계획법, 제약 최적화를 다룬다.

세부 항목은 다음과 같다.

- Nonlinear Equations
- Numerical Linear Algebra
- Interpolation and Approximation
- Numerical Differentiation
- Numerical Integration
- Ordinary Differential Equations
- Partial Differential Equations
- Optimization
- Finite Difference Method (FDM)
- Finite Element Method (FEM)
- Finite Volume Method (FVM)
- Particle Tracing
- Pi Computation
- Chaos and Fractals

### 3.4. [ML] Machine Learning

머신러닝은 데이터로부터 패턴을 학습하여 예측이나 의사결정을 수행하는 방법론을 다룬다. Classification과 Regression은 지도학습의 기본 태스크이며, Clustering은 비지도학습의 대표 방법이다. Time Series Analysis는 시간적 순서를 가진 데이터의 패턴 파악과 예측을 다루며, Anomaly Detection은 정상 패턴에서 벗어난 관측값을 탐지한다.

세부 항목은 다음과 같다.

- Classification
- Regression
- Clustering
- Time Series Analysis
- Anomaly Detection
- AutoML (PyCaret)

### 3.5. [CV] Computer Vision

컴퓨터 비전은 이미지·영상 데이터에서 의미 있는 정보를 추출하는 딥러닝 응용 분야다. Image Preprocessing 단계에서는 OpenCV·torchvision·albumentations 등을 활용하여 정규화, 크기 변환, 데이터 증강을 수행한다.

핵심 태스크는 Image Classification, Image Segmentation, Object Detection으로 구분된다. Vision Anomaly Detection은 정상 이미지와의 편차를 측정하여 표면 결함 등의 이상을 식별하며, Vision Backbones는 ResNet, ViT, EfficientNet 등 특징 추출에 사용되는 사전 학습 모델 구조를 다룬다.

세부 항목은 다음과 같다.

- Image Preprocessing (OpenCV / scikit-image / PIL / torchvision / augmentations)
- Image Classification
- Image Segmentation
- Object Detection
- Vision Anomaly Detection
- Vision Backbones
- Vision Datasets

### 3.6. [NLP] Natural Language Processing

자연어 처리는 텍스트 데이터를 기계가 이해하고 생성할 수 있도록 처리하는 분야다. Text Preprocessing과 Tokenization은 원시 텍스트를 모델 입력 형태로 변환하는 기초 단계이며, Word Embeddings는 단어를 연속 벡터 공간에 표현하는 방법론을 다룬다.

Sequence Modeling에서는 RNN·LSTM 계열을 다루며, Attention Mechanism과 Transformer 아키텍처로 이어진다. 응용 분야로는 Text Classification, Text Generation, Retrieval-Augmented Generation(RAG)이 있다.

세부 항목은 다음과 같다.

- Text Preprocessing
- Tokenization
- Word Embeddings
- Sequence Modeling
- Attention Mechanism
- Transformer
- Language Models
- Text Classification
- Text Generation
- Retrieval-Augmented Generation
- Hugging Face and Open LLM Models

### 3.7. [CS] Computer Science

컴퓨터 과학 분야는 프로그래밍 언어, 자료구조, 알고리즘, 소프트웨어 설계를 다룬다. Python Programming은 이 워크스페이스의 주 언어로 과학 계산·데이터 분석·딥러닝 프레임워크 활용을 포함하며, C++ Programming은 성능이 중요한 시뮬레이션·임베디드 응용에 활용한다.

Data Structures와 Algorithms는 효율적인 프로그램 작성의 기초이며, Design Patterns는 반복되는 소프트웨어 설계 문제에 대한 검증된 해법을 다룬다.

세부 항목은 다음과 같다.

- Python Programming
- C++ Programming
- Data Structures
- Algorithms
- Design Patterns
- Test-Driven Development and Refactoring

### 3.8. [DEV] Development Environment and Tools

개발 환경과 도구는 실제 작업 생산성을 직접 결정하는 분야다. 실행 환경 구성, 버전 관리, 에디터 설정, 문서화 도구, 지식 관리 및 AI 보조 도구의 5개 그룹으로 세분한다.

#### Environment Setup

Anaconda 기반 Python 환경, C++ 빌드 환경, CUDA GPU 환경, WSL 설정, Windows 네이티브 환경 구성을 다룬다.

- Python Environment (Anaconda / conda)
- C++ Environment
- CUDA Environment
- WSL / Anaconda
- Windows Development Environment

#### Version Control

Git 명령·워크플로우와 GitHub 협업 전략(branch strategy, pull request, CI)을 다룬다.

- Git
- GitHub (Development Process / Branch Strategy)

#### Editor and IDE

VSCode 에디터의 확장 구성, 원격 터널(VSCode Tunnel), 단축키 등 개발 환경 최적화를 다룬다.

- VSCode (Tips / Extensions / VSCode Tunnel)

#### Documentation Tools

Jupyter Book v2(MyST Markdown 기반 웹북 생성), Markdown, LaTeX 등 기술 문서 작성 및 배포 도구를 다룬다.

- Jupyter Book (version 1 vs. 2)
- Markdown
- LaTeX

#### Knowledge Management and AI Tools

Obsidian과 PARA / Second Brain 방법론을 통한 지식 체계화, Claude Code 등 AI CLI 보조 도구의 활용과 설정을 다룬다.

- Obsidian
- PARA / Second Brain
- AI CLI Tools

### 3.9. [MISC] Miscellaneous

Miscellaneous는 특정 분류에 귀속되지 않거나 아직 분류가 확정되지 않은 자료를 임시로 보관하는 공간이다. 두 개 이상의 분야에 걸치는 cross-domain 아이디어, 작업 중인 초안, 개인 메모가 여기에 포함된다.

이 분류는 정식 분류가 결정될 때까지 자료의 위치를 보존하는 임시 저장소 역할을 하며, 일정 기간 후 적절한 분류로 이동하거나 삭제 여부를 검토한다.

세부 항목은 다음과 같다.

- Temporary Notes
- Uncategorized Materials
- Cross-domain Ideas
- Personal Memos
- Reference Links
- Drafts
- Legacy Materials
