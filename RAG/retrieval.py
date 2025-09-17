import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import os
from tqdm import tqdm # 긴 작업의 진행률을 표시하기 위한 라이브러리

# --- 설정 (Paths and Model Configuration) ---

# 데이터셋 및 모델 가중치 경로 설정
BASE_PATH = "/home/infidea/rebirth-hjun/KRAG_2025_2"
DATASET_PATH = os.path.join(BASE_PATH, "KRAG_2025/Dataset")
MODEL_WEIGHTS_PATH = os.path.join("/home/infidea/rebirth-hjun", "meeting_model_weights/kanana-1.5-8b-instruct-2505")

# 입력 파일 경로
TRAIN_FILE_PATH = os.path.join(DATASET_PATH, "korean_language_rag_V1.0_train.json")
DEV_FILE_PATH = os.path.join(DATASET_PATH, "korean_language_rag_V1.0_dev.json")
TEST_FILE_PATH = os.path.join(DATASET_PATH, "korean_language_rag_V1.0_test.json")

# 출력 CSV 파일 경로
OUTPUT_CSV_PATH = os.path.join(BASE_PATH, "KRAG_2025/0715_kanana-1.5-8b-instruct-2505_generated_queries_train_dev_test.csv")

# 모델 생성 관련 하이퍼파라미터
GENERATION_CONFIG = {
    "max_new_tokens": 4098,
    "do_sample": False,
    # "temperature": 0.7,
    # "top_p": 0.9,
    # "repetition_penalty": 1.1,
}

# --- 프롬프트 (Instruction Prompt) ---

# 모델에 역할을 부여하고 작업을 지시하는 프롬프트
INSTRUCTION = """# [역할] 당신은 한국어 어문 규범 전문가이자, 검색 증강 생성(RAG) 시스템의 검색 성능을 최적화하는 '쿼리 재작성 전문 AI'입니다. 당신의 주요 임무는 사용자의 자연어 질문을 분석하여, '국어 지식 자료' DB에서 정답의 근거가 되는 규칙을 가장 효과적으로 찾아낼 수 있는 하나의 검색용 쿼리들을 생성하는 것입니다. # [과업] 사용자의 질문이 입력되면, 다음 원칙에 따라 검색에 최적화된 쿼리들을 생성해 주세요. 1. **핵심어 추출**: 질문에서 핵심이 되는 단어, 형태소, 맞춤법 오류 등을 정확히 파악합니다. 2. **규칙 일반화**: 질문의 구체적인 사례를 포괄할 수 있는 일반적인 '어문 규범'이나 '문법 용어'를 떠올립니다. (예: 두음 법칙, 어미 활용, 피동 표현, 표준어 규정 등) 3. **생성 규칙**: 쿼리는 검색 효율성을 위해 간결해야 합니다. # [예시] ## 예시 1 - 입력: "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요. - 출력: ["의존명사 양 량 두음법칙"] ## 예시 2 - 입력: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그렇게 고친 이유를 설명하세요. "어서 쾌차하시길 바래요." - 출력: ["동사 '바라다' 어미 '-아요' 결합"] ## 예시 3 - 입력: "이 자리를 빌어 감사의 말씀을 전합니다."에서 '빌어'가 맞나요, '빌려'가 맞나요? - 출력: ["'빌다'와 '빌리다'의 의미 차이"] # [출력 형식] - 생성된 쿼리들은 반드시 strings으로로만 출력해 주세요. - 다른 설명은 일절 포함하지 마세요. # [실행] 이제 아래 사용자의 질문에 대한 검색용 쿼리들을 생성해 주세요. - 입력: """

# --- 함수 정의 (Function Definitions) ---

def read_json_file(file_path):
    """지정된 경로의 JSON 파일을 읽어 데이터를 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"성공적으로 파일을 읽었습니다: {file_path}")
        return data
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"오류: JSON 디코딩 오류 - {file_path}")
        return None
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생 {file_path}: {e}")
        return None

def save_to_csv(data, output_path):
    """주어진 데이터를 CSV 파일로 저장합니다."""
    if not data:
        print("CSV로 저장할 데이터가 없습니다.")
        return

    try:
        # 출력 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            # CSV 파일의 헤더(열 이름)를 정의합니다.
            # 데이터의 첫 번째 항목의 키를 기반으로 헤더를 동적으로 생성할 수 있습니다.
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(data)
        print(f"결과를 성공적으로 CSV 파일에 저장했습니다: {output_path}")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")


# --- 메인 실행 로직 (Main Execution Logic) ---

def main():
    """메인 로직을 실행하는 함수"""
    
    # 1. 데이터 로드
    train_json_data = read_json_file(TRAIN_FILE_PATH)
    dev_json_data = read_json_file(DEV_FILE_PATH)
    test_json_data = read_json_file(TEST_FILE_PATH)
    

    datasets_to_combine = []
    if train_json_data:
        datasets_to_combine.extend(train_json_data)
    if dev_json_data:
        datasets_to_combine.extend(dev_json_data)
    if test_json_data:
        datasets_to_combine.extend(test_json_data)
        
    # 합쳐진 데이터셋을 'combined_data' 변수에 할당
    combined_data = datasets_to_combine
    
    if not train_json_data:
        # 파일 로드 실패 시 프로그램 종료
        return

    # 2. 모델 및 토크나이저 로드
    print("모델과 토크나이저를 로드하는 중입니다...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_WEIGHTS_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_WEIGHTS_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # 생성 관련 설정에 eos_token_id 추가
        GENERATION_CONFIG['eos_token_id'] = tokenizer.eos_token_id

    except Exception as e:
        print(f"모델 또는 토크나이저 로드 중 오류 발생: {e}")
        return
    print("모델과 토크나이저 로드가 완료되었습니다.")

    # 3. 쿼리 생성 및 결과 저장
    results_for_csv = []

    print("입력 데이터에 대한 쿼리 생성을 시작합니다...")
    # tqdm을 사용하여 진행률 표시
    for item in tqdm(combined_data, desc="쿼리 생성 중"):
        question = item['input']['question']
        
        # 모델 입력 형식 구성
        messages = [
            {"role": "user", "content": INSTRUCTION + f"\"{question}\""}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 텍스트 생성
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG
        )
        
        # 생성된 결과 디코딩
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # CSV에 저장할 데이터 구성
        # 원본 데이터와 생성된 쿼리를 함께 저장
        result_item = {
            'id': item.get('id', ''),
            'question': question,
            'expected_output': item.get('output', ''),
            'generated_queries': generated_text.strip() # 생성된 결과의 앞뒤 공백 제거
        }
        results_for_csv.append(result_item)

    # 4. CSV 파일로 결과 저장
    save_to_csv(results_for_csv, OUTPUT_CSV_PATH)
    print("모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()