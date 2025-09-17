import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, # 학습 설정을 위한 클래스
    # Trainer, # [수정] SFTTrainer를 사용하므로 제거
)

# [수정] SFTTrainer와 관련 클래스 import
from trl import SFTConfig, SFTTrainer

# [수정] 응답 부분만 학습하기 위한 데이터 콜레이터 import
from trl import DataCollatorForCompletionOnlyLM

from peft import LoraConfig, get_peft_model 

import wandb

from datasets import Dataset # Hugging Face 데이터셋 객체로 변환하기 위함
import os
import pandas as pd
import numpy as np
from datetime import datetime # 날짜와 시간을 가져오기 위해 추가
import pytz # 한국 시간대를 설정하기 위해 추가

def read_json_file(file_path):
    """지정된 경로의 JSON 파일을 읽어 데이터를 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"오류: JSON 디코딩 오류 - {file_path}")
        return None
    except Exception as e:
        print(f"읽는 중 오류 발생 {file_path}: {e}")
        return None

def main():
    """메인 학습 실행 함수"""

    # --- 1. 설정 (Configuration) ---
    # 기반으로 사용할 모델 (fine-tuning의 시작점)
    
    # 학습 데이터 파일 경로
    train_file_path = "/home/infidea/rebirth-hjun/KRAG_2025/pre/train.csv"
    train_dev_test_file_path1 = "/home/infidea/rebirth-hjun/KRAG_2025/pre/0715-train_dev_test.csv" #경로 수정

    # 학습된 모델과 체크포인트가 저장될 디렉터리 (기존 코드에서 가져옴)
    output_dir = "/home/infidea/rebirth-hjun/KRAG_2025/trained_model"

    model_name = "/home/infidea/rebirth-hjun/meeting_model_weights/kanana-1.5-8b-instruct-2505"
    epoch = 6

    # --- 2. 데이터 로드 및 전처리 (Data Loading & Preprocessing) ---

    print("학습 데이터를 로드하고 전처리를 시작합니다...")

    # 첫 번째 CSV 데이터 로드
    df1 = pd.read_csv(train_dev_test_file_path1)
    print(f"'{train_dev_test_file_path1}' 파일 로드 완료. 첫 1행:")
    print(df1.head(1))

    # 두 DataFrame을 합치기 (concat)
    # ignore_index=True를 사용하면 합쳐진 DataFrame의 인덱스가 0부터 새로 시작합니다.
    train_data_df = pd.concat([df1], ignore_index=True)
    train_data_df = train_data_df.dropna()

    print(f"\n두 파일을 합친 후 전체 데이터 크기: {len(train_data_df)} 행")
    print(train_data_df.head())

    # 기존 코드의 샘플링 로직 (필요하다면 유지)
    train_data_df = train_data_df.sample(frac=1.0, random_state=42).copy()
        
    # [수정] SFTTrainer의 채팅 형식에 맞게 데이터를 변환하는 함수
    def create_chat_format(row):
        return [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": row['answer']}
        ]

    # 'messages' 컬럼 추가
    train_data_df['messages'] = train_data_df.apply(create_chat_format, axis=1)
    
    # Hugging Face Dataset 객체로 변환
    train_dataset = Dataset.from_pandas(train_data_df)

    # 토크나이저 로드 (전처리에 필요)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Qwen3 모델은 기본적으로 pad_token이 없습니다. 
    # 패딩 토큰을 EOS(End Of Sentence) 토큰으로 설정해야 학습 시 배치 처리가 원활합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- [수정된 코드] formatted_prompt 토큰 길이 분포 확인 ---
    print("\n--- 토큰 길이 분포 확인 시작 ---")

    # SFTTrainer가 내부적으로 수행할 포맷팅을 미리 적용하여 길이를 계산
    def get_token_length(example):
        formatted_prompt = tokenizer.apply_chat_template(
            example['messages'], 
            tokenize=False,
            add_generation_prompt=False
        )
        return {'token_length': len(tokenizer(formatted_prompt).input_ids)}

    # 토큰 길이 계산 및 통계 출력
    token_lengths_dataset = train_dataset.map(get_token_length)
    token_lengths = token_lengths_dataset['token_length']
    
    print("토큰 길이 통계:")
    print(f"  - 전체 샘플 수: {len(token_lengths)}")
    print(f"  - 최소 길이: {min(token_lengths)}")
    print(f"  - 최대 길이: {max(token_lengths)}")
    print(f"  - 평균 길이: {np.mean(token_lengths):.2f}")
    print(f"  - 95% 백분위수 길이: {np.percentile(token_lengths, 95):.2f}")
    print("--- 토큰 길이 분포 확인 완료 ---\n")
    
    # --- 3. 모델 로드 (Model Loading) ---
    print(f"'{model_name}'에서 모델을 로드합니다...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto", # bfloat16 또는 float16 자동 선택
            device_map="auto",   # 사용 가능한 GPU에 모델 레이어 자동 분배
        )
        print("모델 로드 완료.")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # --- 4. 트레이너 설정 (Trainer Setup) ---
    print("학습 설정을 구성합니다.")

    # [수정] LoRA 설정 활성화
    lora_config = LoraConfig(
        r=32,  # LoRA rank
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,  # LoRA alpha
        lora_dropout=0,  # LoRA dropout
        bias="none",  # 바이어스 설정
        task_type="CAUSAL_LM",  # 태스크 타입
    )

    # [수정] PEFT 모델 생성
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # TrainingArguments: 학습에 필요한 모든 하이퍼파라미터를 정의
    sft_config = SFTConfig(
        # --- 기본 학습 인자 ---
        output_dir=output_dir,
        num_train_epochs=epoch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        # bf16=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_seq_length=2048,
        lr_scheduler_kwargs={"num_cycles":3},
        )

    # [수정] 응답 템플릿만 학습하기 위한 Data Collator 정의
    # Qwen2 모델의 어시스턴트 시작 템플릿을 지정합니다.
    # instruction_template = "<|start_header_id|>user<|end_header_id|>"
    if "kanana" in model_name:
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
    elif "A.X" in model_name:
        response_template = "<|im_end|><|im_start|><|assistant|>"
    elif "EXAONE" in model_name:
        response_template = "[|assistant|]"  
    elif "Qwen" in model_name:
        response_template = "<|im_start|>assistant"
    elif "gemma" in model_name:
        response_template = "<start_of_turn>model"

    collator = DataCollatorForCompletionOnlyLM(
        # instruction_template= instruction_template,
        response_template = response_template,
        tokenizer=tokenizer,
    )
        
    # [수정] Trainer를 SFTTrainer로 교체
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # --- 5. 학습 실행 (Run Training) ---
    print("모델 학습을 시작합니다...")
    trainer.train()
    print("학습 완료.")

    # 한국 시간대(KST)를 기준으로 현재 시간 가져오기
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    timestamp = now_kst.strftime("%Y%m%d-%H%M") # 예: '20250702-1435' 형식

    # model_name 경로에서 마지막 디렉터리 이름(모델 이름) 추출
    base_model_name = os.path.basename(model_name.strip('/'))

    # 새로운 폴더 이름 생성: 모델이름-타임스탬프
    final_folder_name = f"{base_model_name}-lora-{timestamp}"
    
    # 최종 저장 경로 조합
    final_model_path = os.path.join(output_dir, final_folder_name)
    
    # --- 6. 최종 모델 저장 ---
    print(f"학습된 최종 어댑터(모델)를 '{final_model_path}' 경로에 저장합니다.")
    trainer.save_model(final_model_path)
    print("최종 모델 저장 완료.")


if __name__ == "__main__":
    main()
