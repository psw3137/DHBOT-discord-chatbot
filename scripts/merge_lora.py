"""
LoRA 어댑터 머지 스크립트
- 베이스 모델 + LoRA 어댑터 → 단일 모델 (models/friend-merged/)
- Ollama에 등록하기 위한 사전 작업
실행: py -3.14 scripts/merge_lora.py
"""

import sys
import os
import torch

sys.stdout.reconfigure(encoding='utf-8')

LORA_PATH = os.path.join(os.path.dirname(__file__), '../models/friend-lora')
MERGED_PATH = os.path.join(os.path.dirname(__file__), '../models/friend-merged')
BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print('베이스 모델 로드 중 (16-bit)...')
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True,
    )

    print('LoRA 어댑터 로드 및 머지 중...')
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()

    print(f'머지 완료. 저장 중: {MERGED_PATH}')
    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    print('저장 완료.')

if __name__ == '__main__':
    main()
