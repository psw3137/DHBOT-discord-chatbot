"""
파인튜닝 모델 추론 서버
- 포트 8002에서 HTTP 요청 수신
- LoRA 어댑터 + 4-bit 양자화로 메모리 효율적 로드
- 요청: {"messages": [...], "max_new_tokens": 200}
- 응답: {"response": "..."}

실행: py -3.14 scripts/inference_server.py
"""

import sys
import os
import json
import torch
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
LORA_PATH = os.path.join(os.path.dirname(__file__), '../models/friend-lora')
PORT = 8002
MAX_NEW_TOKENS = 200


print('모델 로드 중...')
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={'': 0},
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH, device_map={'': 0})
model.eval()
print(f'모델 로드 완료 → http://localhost:{PORT}')


def generate(messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 부분 제거하고 생성된 텍스트만 반환
    generated = output[0][inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length).decode('utf-8'))
        messages = body.get('messages', [])
        max_new_tokens = body.get('max_new_tokens', MAX_NEW_TOKENS)

        try:
            response_text = generate(messages, max_new_tokens)
            result = {'response': response_text}
            status = 200
        except Exception as e:
            result = {'error': str(e)}
            status = 500

        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    HTTPServer(('localhost', PORT), InferenceHandler).serve_forever()
