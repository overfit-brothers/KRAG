import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import csv
import os
import glob
from tqdm import tqdm
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import random
from datetime import datetime
import pytz
from huggingface_hub import HfApi
import gc

api = HfApi()

repo_id = "hyoungjoon/krag" # 예: "your-username/my-awesome-model"
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)


# --- 설정 (Configuration) ---
# 경로, 모델명 등 주요 설정을 상수로 관리하여 변경이 용이하도록 함
BASE_PATH = "/home/infidea/rebirth-hjun"
DATASET_PATH = os.path.join(BASE_PATH, "KRAG_2025/Dataset")
DATA_PATH = os.path.join(BASE_PATH, "KRAG_2025/pre")
MODEL_WEIGHTS_PATH = "/home/infidea/rebirth-hjun/KRAG_2025/modle_merge/69.94/"
MODEL_LORA_WEIGHTS_PATH = "/home/infidea/rebirth-hjun/KRAG_2025/trained_model/kanana-1.5-8b-instruct-2505-lora-20250719-1945"
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"
EMBEDDINGS_TENSOR_PATH = os.path.join(BASE_PATH, "KRAG_2025/RAG/all_embeddings_tensor.pt")
OUTPUT_JSON_PATH = os.path.join(BASE_PATH, "KRAG_2025/inference/2025/test_optimized.json")

# 쿼리 생성 모델 하이퍼파라미터
QUERY_GEN_CONFIG = {
    "max_new_tokens": 32,
    "do_sample": True,
}

# 최종 답변 생성 모델 하이퍼파라미터
ANSWER_GEN_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": False,
}

# Few-shot 예제를 가져올 데이터셋 범위
FEW_SHOT_EXAMPLE_RANGE = 749

# 프롬프트
INSTRUCTION_PROMPT = """# [역할] 당신은 한국어 어문 규범 전문가이자, 검색 증강 생성(RAG) 시스템의 검색 성능을 최적화하는 '쿼리 재작성 전문 AI'입니다. 당신의 주요 임무는 사용자의 자연어 질문을 분석하여, '국어 지식 자료' DB에서 정답의 근거가 되는 규칙을 가장 효과적으로 찾아낼 수 있는 하나의 검색용 쿼리들을 생성하는 것입니다. # [과업] 사용자의 질문이 입력되면, 다음 원칙에 따라 검색에 최적화된 쿼리들을 생성해 주세요. 1. **핵심어 추출**: 질문에서 핵심이 되는 단어, 형태소, 맞춤법 오류 등을 정확히 파악합니다. 2. **규칙 일반화**: 질문의 구체적인 사례를 포괄할 수 있는 일반적인 '어문 규범'이나 '문법 용어'를 떠올립니다. (예: 두음 법칙, 어미 활용, 피동 표현, 표준어 규정 등) 3. **생성 규칙**: 쿼리는 검색 효율성을 위해 간결해야 합니다. # [예시] ## 예시 1 - 입력: "가축을 기를 때에는 {먹이량/먹이양}을 조절해 주어야 한다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요. - 출력: ["의존명사 양 량 두음법칙"] ## 예시 2 - 입력: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그렇게 고친 이유를 설명하세요. "어서 쾌차하시길 바래요." - 출력: ["동사 '바라다' 어미 '-아요' 결합"] ## 예시 3 - 입력: "이 자리를 빌어 감사의 말씀을 전합니다."에서 '빌어'가 맞나요, '빌려'가 맞나요? - 출력: ["'빌다'와 '빌리다'의 의미 차이"] # [출력 형식] - 생성된 쿼리들은 반드시 strings으로로만 출력해 주세요. - 다른 설명은 일절 포함하지 마세요. # [실행] 이제 아래 사용자의 질문에 대한 검색용 쿼리들을 생성해 주세요. - 입력: """

def load_datasets():
    """train, dev, test JSON 파일을 읽어 하나의 데이터프레임으로 결합합니다."""
    print("데이터셋을 로드합니다...")
    paths = {
        "train": os.path.join(DATASET_PATH, "korean_language_rag_V1.0_train.json"),
        "dev": os.path.join(DATASET_PATH, "korean_language_rag_V1.0_dev.json"),
        "test": os.path.join(DATASET_PATH, "korean_language_rag_V1.0_test.json")
    }
    
    data_frames = []
    for key, path in paths.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_frames.append(pd.DataFrame(data))
                print(f"성공적으로 파일을 읽었습니다: {path}")
        except Exception as e:
            print(f"{path} 파일 로딩 중 오류 발생: {e}")

    if not data_frames:
        return None
        
    combined_df = pd.concat(data_frames, ignore_index=True)
    print("모든 데이터셋을 성공적으로 결합했습니다.")
    return combined_df


def load_models(model_weights_path, embedding_model_name, adapter_path):
    """LLM과 임베딩 모델을 로드합니다."""
    print("모델과 토크나이저를 로드하는 중입니다...")
    try:
        print(MODEL_WEIGHTS_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_weights_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_weights_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        embedding_model = SentenceTransformer(embedding_model_name)
        
        # model = PeftModel.from_pretrained(model, adapter_path)
        # model = model.merge_and_unload()
        
        print("모델과 토크나이저 로드가 완료되었습니다.")
        return model, tokenizer, embedding_model
    except Exception as e:
        print(f"모델 또는 토크나이저 로드 중 오류 발생: {e}")
        return None, None, None


def setup_rag_pipeline(data_path, embedding_model_name):
    """
    [핵심 개선사항] RAG 파이프라인(문서 로딩, 분할, 벡터 저장소 생성)을 미리 설정합니다.
    이 함수는 한 번만 호출되어야 합니다.
    """
    print("\n--- RAG 파이프라인 설정 시작 ---")
    try:
        # 1. 문서 로드
        loader = DirectoryLoader(
            data_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()
        if not documents:
            print(f"오류: '{data_path}'에서 문서를 찾을 수 없습니다.")
            return None
        print(f"총 {len(documents)}개의 문서를 로드했습니다.")

        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
        texts = text_splitter.split_documents(documents)
        print(f"문서를 {len(texts)}개의 텍스트 조각으로 분할했습니다.")

        # 3. 임베딩 모델 설정 및 벡터 저장소 생성
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        print("--- RAG 파이프라인 설정 완료 ---")
        return retriever
    except Exception as e:
        print(f"RAG 파이프라인 설정 중 오류 발생: {e}")
        return None


def save_results_to_json(data, base_file_path):
    """
    결과 데이터를 지정된 경로의 JSON 파일로 저장합니다.
    파일 이름에 현재 한국 시간 날짜-시-분 정보를 추가합니다.
    """
    try:
        # 한국 시간대 설정
        korea_tz = pytz.timezone('Asia/Seoul')
        now = datetime.now(korea_tz)

        # 현재 시간을 'YYYY-MM-DD_HH-MM' 형식으로 포맷팅합니다.
        timestamp = now.strftime('%Y-%m-%d_%H-%M')

        # 파일 경로와 파일명을 분리하고, 시간 정보를 파일명에 추가합니다.
        directory, filename = os.path.split(base_file_path)
        name, ext = os.path.splitext(filename)
        
        # 새로운 파일명 생성: '원본파일명_YYYY-MM-DD_HH-MM.json'
        new_filename = f"{name}_{timestamp}{ext}"
        file_path = os.path.join(directory, new_filename)

        # 파일 경로의 디렉터리가 존재하지 않으면 생성합니다.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"결과를 '{file_path}' 파일에 성공적으로 저장했습니다.")
    except Exception as e:
        print(f"결과 파일 저장 중 오류 발생: {e}")


def main():
    """메인 로직을 실행하는 함수"""
    # 1. 모델 및 RAG 파이프라인 사전 로드
    
    model_files = glob.glob(os.path.join(MODEL_WEIGHTS_PATH, '*'))
    for i in model_files:
        print(i)
    
        model, tokenizer, embedding_model = load_models(i, EMBEDDING_MODEL_NAME, MODEL_LORA_WEIGHTS_PATH)
        retriever = setup_rag_pipeline(DATA_PATH, EMBEDDING_MODEL_NAME)
        
        if not all([model, tokenizer, embedding_model, retriever]):
            print("초기 설정 실패. 프로그램을 종료합니다.")
            return

        # 2. 데이터셋 로드 및 준비
        df_combined = load_datasets()
        if df_combined is None:
            print("데이터셋 로딩 실패. 프로그램을 종료합니다.")
            return

        # Few-shot 예제용 임베딩 텐서 로드 및 슬라이싱 (미리 준비)
        all_embeddings_tensor = torch.load(EMBEDDINGS_TENSOR_PATH).to('cuda')
        few_shot_embeddings = all_embeddings_tensor[:FEW_SHOT_EXAMPLE_RANGE]
        
        # 처리할 데이터 슬라이싱
        inference_data = df_combined.iloc[FEW_SHOT_EXAMPLE_RANGE:].to_dict('records')
        # inference_data = df_combined.iloc[:FEW_SHOT_EXAMPLE_RANGE].to_dict('records')
        
        results_for_csv = []
        final_prompts = []
        df_new = pd.DataFrame()
        # 3. 메인 추론 루프
        for n , item in tqdm(enumerate(inference_data), desc="전체 데이터 처리 중", total=len(inference_data)):
            question = item['input']['question']
            
            # 3-1. 검색 쿼리 생성
            messages = [{"role": "user", "content": INSTRUCTION_PROMPT + f"\"{question}\""}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            query_outputs = model.generate(**inputs, **QUERY_GEN_CONFIG, eos_token_id=tokenizer.eos_token_id)
            generated_query = tokenizer.decode(query_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # 3-2. RAG를 이용한 문서 검색
            retrieved_docs = retriever.get_relevant_documents(generated_query)
            retrieved_content = retrieved_docs[0].page_content if retrieved_docs else "검색된 내용이 없습니다."
            
            # 3-3. Few-shot 예제 검색
            current_q_type = item['input']['question_type']
            query_embedding = embedding_model.encode(generated_query, convert_to_tensor=True)
            
            # 동일한 질문 유형을 가진 예제만 필터링
            type_mask = df_combined['input'][:FEW_SHOT_EXAMPLE_RANGE].apply(lambda x: x['question_type'] == current_q_type)
            candidate_indices = type_mask[type_mask].index.tolist()
            
            if candidate_indices:
                candidate_embeddings = few_shot_embeddings[candidate_indices]
                
                # 최적화: 필요한 후보군에 대해서만 유사도 계산
                cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
                
                # 상위 k개 선택 (후보가 5개 미만일 수 있으므로 min 사용)
                k_value = min(5, len(candidate_indices))
                top_results = torch.topk(cosine_scores, k=k_value)
                
                # 원래 인덱스로 변환
                similar_original_indices = [candidate_indices[i] for i in top_results.indices]
                
                # 상위 k개 중 1개 랜덤 샘플링
                selected_index = random.choice(similar_original_indices)
                
                few_shot_example = f"question:{df_combined['input'][selected_index]['question']}\nanswer:{df_combined['output'][selected_index]['answer']}"
            else:
                few_shot_example = "" # 유사한 예제가 없는 경우

            # 3-4. 최종 프롬프트 구성 및 답변 생성
            final_prompt_template = (
                "다음은 어문 규범에 대한 자료입니다.\n\n"
                f"{retrieved_content}\n\n"
                "아래는 질문과 답변의 예시입니다. 이 예시의 형식에 맞춰 질문에 답해주세요.\n\n"
                f"{few_shot_example}\n\n"
                "**위 'answer:' 뒤의 출력 형식을 엄격히 준수하여 답변을 생성하십시오.**\n"
                "\n이제 다음 질문에 답변하세요.\n"
                f"question:{question}"
            )
            final_prompts.append(final_prompt_template)
            
            messages = [{"role": "user", "content": final_prompt_template}]
            final_inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            
            output_ids = model.generate(
                final_inputs.to("cuda"), # 모델이 GPU에 있다면 .to("cuda") 유지
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False, # 샘플링을 위해 do_sample=True를 추가하는 것이 좋습니다.
            )
            
            prompt_length = final_inputs.shape[1]
            
            generated_only_text = tokenizer.decode(
                output_ids[0][prompt_length:], 
                skip_special_tokens=True
            )
            # if "가 옳다." not in generated_only_text:
            #     generated_only_text.split(".\"")
            results_for_csv.append({
                "id": item['id'],
                "input": item['input']['question'],
                "output": {"answer": generated_only_text.replace("</think>","").replace("으가","가").replace("로 고쳐 쓴다.","가 옳다.").replace("<think>\n\n\n\n\\","").strip()}
            })
            
            if n == 100: # 5번째 반복 후
                print(f"\n[알림] 5번째 반복 완료, 중간 결과를 저장합니다.")
                # 중간 저장 파일 이름에도 표시를 추가할 수 있습니다.
                intermediate_file_path = OUTPUT_JSON_PATH.replace(".json", "_intermediate.json")
                save_results_to_json(results_for_csv, intermediate_file_path)
                
        # df_new['input'] = final_prompts
        # df_new.to_csv('final.csv', index=False, encoding='utf-8-sig')
        
        # 4. CSV 파일로 최종 결과 저장
        save_results_to_json(results_for_csv, OUTPUT_JSON_PATH)
        print("\n모든 작업이 완료되었습니다.")
        
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("--- 리소스 정리 완료 ---")
        
    # 업로드할 로컬 폴더 경로
    # local_folder_path = "/home/infidea/rebirth-hjun/KRAG_2025/inference/2025" # 예: "./my_model"
    # repo_id = "사용자이름/모델_이름"

    # api.upload_folder(
    #     folder_path=local_folder_path,
    #     repo_id=repo_id,
    #     commit_message="Initial model upload" # 커밋 메시지
    # )


if __name__ == "__main__":
    main()