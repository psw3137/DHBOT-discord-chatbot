"""
QLoRA 파인튜닝 스크립트
- 베이스 모델: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
- 방법: QLoRA (4-bit 양자화 + LoRA)
- 데이터: data/finetune_data.jsonl
- 출력: models/friend-lora/

실행: python scripts/finetune.py
예상 시간: RTX 3080 10GB 기준 6~12시간
"""

import os
import sys
import json
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/finetune_data.jsonl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../models/friend-lora')
MAX_SEQ_LENGTH = 512

# LoRA 설정
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                  'gate_proj', 'up_proj', 'down_proj']

# 학습 설정 (RTX 3080 10GB 기준)
BATCH_SIZE = 1
GRAD_ACCUM = 8          # 실질 배치 = 8
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS = 200
LOGGING_STEPS = 20


def load_dataset():
    with open(DATA_PATH, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f'데이터 로드: {len(records)}개')
    return records


def format_chat(record, tokenizer):
    """messages 리스트를 모델 chat template으로 변환"""
    return tokenizer.apply_chat_template(
        record['messages'],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    print(f'CUDA: {torch.cuda.is_available()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # ── 4-bit 양자화 설정 ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 토크나이저 로드 ──
    print(f'\n모델 로드 중: {BASE_MODEL}')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # ── 모델 로드 (4-bit) ──
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA 적용 ──
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 데이터셋 준비 ──
    records = load_dataset()
    formatted = [format_chat(r, tokenizer) for r in records]
    dataset = Dataset.from_dict({'text': formatted})

    print(f'\n학습 시작')
    print(f'배치 크기: {BATCH_SIZE} × 그래디언트 누적 {GRAD_ACCUM} = 실질 {BATCH_SIZE * GRAD_ACCUM}')
    print(f'에폭: {EPOCHS}, LR: {LR}')

    # ── 학습 설정 (TRL 0.29+ SFTConfig 사용) ──
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=int(EPOCHS * len(dataset) / (BATCH_SIZE * GRAD_ACCUM) * 0.05),
        lr_scheduler_type='cosine',
        fp16=False,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        optim='paged_adamw_8bit',
        report_to='none',
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field='text',
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=True)

    print(f'\nLoRA 어댑터 저장: {OUTPUT_DIR}')
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print('완료.')


if __name__ == '__main__':
    main()
