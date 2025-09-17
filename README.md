# KRAG_2025 (2025년 한국어 어문 규범 기반 생성)

> **[2025 한국어 어문 규범 기반 생성(RAG)(가 유형) 공식 링크]**(https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=182&clCd=ING_TASK&subMenuId=sub01)

## EDA (Exploratory Data Analysis)

* `train` 데이터셋에 대한 탐색적 데이터 분석을 수행합니다.

## 주요 기능 및 코드 설명

### 1. 데이터 전처리 및 RAG

* **`pre/make_pdf2txt.ipynb`**
    * 제공된 `국어 지식 기반 생성(RAG) 참조 문서.pdf` 파일을 RAG에 활용하기 용이한 `txt` 파일 형식으로 변환합니다.

* **`RAG/retrieval.py`**
    * 사용자의 질문(query)을 임베딩 모델이 이해하고 검색하기 적합한 형태로 가공 및 변환합니다.

* **`RAG/inference.ipynb`**
    * 변환된 쿼리를 기반으로 Vector DB에서 관련성이 높은 문서를 검색합니다.
    * Few-shot 학습을 위해 현재 질문과 가장 유사한 질문-답변 쌍(n-shot)을 함께 가져와 데이터셋을 구성합니다.

### 2. 모델 학습 (SFT)

* **`train/train_SFT.py`**
    * 앞선 전처리 및 RAG 과정을 통해 생성된 데이터셋을 사용하여 언어 모델을 파인튜닝(Supervised Fine-Tuning)합니다.

### 3. 모델 병합

* **`modle_merge/Merge.ipynb`**
    * 학습된 모델(LoRA 어댑터 등)을 원본 모델과 병합합니다.
    * 모델 병합에 대한 상세 설정은 `config.yaml` 파일에서 관리합니다.

### 4. 추론

* **`inference/final.py`**
    * 실제 서비스 단계에서 사용자의 입력을 받아 답변을 생성하는 추론 코드입니다.
    * 입력 데이터를 학습 시와 동일한 전처리 및 RAG 과정을 거쳐 모델의 입력 형식에 맞게 처리한 후, 최종 답변을 출력합니다.