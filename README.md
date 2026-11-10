# DHBOT — 친구 말투 모방 디스코드 봇

친구 한 명의 디스코드 채팅 기록(약 2년치)으로 LLM을 QLoRA 파인튜닝해서, 그 친구의 말투로 대답하는 디스코드 봇입니다.

> 개인 채팅 데이터, 학습된 모델, 대화 로그는 개인정보 보호를 위해 레포에 포함하지 않았습니다. 직접 쓰려면 본인 서버의 채팅을 내보내서 다시 학습해야 합니다.

## 동작 방식

```
디스코드에서 봇 멘션
   └─ src/index.js        봇 진입점
   └─ src/handler.js      멘션 감지, 유저별 대화 맥락, 로그, !초기화 명령
   └─ src/local_model.js  추론 서버로 HTTP 요청
        └─ scripts/inference_server.py  (localhost:8002)
             Qwen2.5-7B-Instruct (4-bit) + LoRA 어댑터로 응답 생성
```

## 학습 파이프라인

| 단계 | 스크립트 | 입력 → 출력 |
| --- | --- | --- |
| 1. 데이터 준비 | `scripts/prepare_finetune.py` | `data/raw/*.json` → `data/finetune_data.jsonl` |
| 2. QLoRA 학습 | `scripts/finetune.py` | → `models/friend-lora/` |
| 3. (선택) 머지 | `scripts/merge_lora.py` | → `models/friend-merged/` |

- 원본 데이터는 [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter)의 JSON 형식입니다. 채널별로 `data/raw/`에 넣습니다.
- 대상 친구가 말하기 직전의 메시지(최대 15개)를 맥락으로, 그 친구의 발화를 정답으로 하는 학습 예제를 만듭니다.
- 같은 사람이 5분 안에 이어서 보낸 메시지는 하나로 묶습니다.
- `TTS` 채널은 음성을 텍스트로 바꾼 것이라 맥락을 믿기 어려워서, 친구의 발화만 말투 학습용으로 씁니다.

학습 설정: 4-bit nf4 양자화, LoRA r=16 / alpha=32, max_length 512, 실질 배치 8, 3 에폭, LR 2e-4. RTX 3080 10GB 기준 약 5시간 걸렸습니다(13,236개 예제).

## 실행 방법

### 요구 사항

- Node.js 20+
- Python 3.10+, CUDA를 쓸 수 있는 NVIDIA GPU (VRAM 10GB 이상)

### 설치

```bash
npm install
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes datasets accelerate
```

### 환경 변수

`.env.example`을 복사해서 `.env`를 만들고 값을 채웁니다.

```env
DISCORD_TOKEN=디스코드_봇_토큰
FRIEND_NAME=흉내낼_친구의_디스코드_닉네임
CONTEXT_LENGTH=0
```

`GROQ_*`, `CHROMA_HOST`, `RAG_TOP_K`는 예전 RAG 버전에서만 씁니다.

### 학습

```bash
python scripts/prepare_finetune.py
python scripts/finetune.py
```

`finetune.py`는 `resume_from_checkpoint=True`로 설정돼 있습니다. 처음 학습할 때는 `False`로 바꿔야 합니다.

### 봇 실행

```bash
# 터미널 1: 추론 서버 (모델 로드에 1~2분)
python scripts/inference_server.py

# 터미널 2: 봇
node src/index.js
```

디스코드에서 `@봇 메시지`로 말을 겁니다. `@봇 !초기화`를 보내면 본인의 대화 맥락이 초기화됩니다.

## 결과와 알려진 문제

말투와 어휘(짧은 반말, 욕설, 가챠겜 용어)는 꽤 잘 따라 했지만, 질문과 상관없는 대답을 하는 경우가 많았습니다. 코드를 다시 분석해 보니 원인으로 추정되는 것은 다음과 같습니다.

1. **학습과 추론의 입력 형식이 다릅니다.** 학습 데이터에는 시스템 프롬프트와 `[닉네임]: 내용` 형식의 채팅 로그가 들어갔는데, 추론할 때는 시스템 프롬프트 없이 메시지 한 줄만 보냅니다.
2. **맥락까지 전부 loss에 포함됐습니다.** 정답(친구의 발화)만이 아니라 다른 사람들의 채팅까지 예측하도록 학습됐습니다.
3. **맥락에 다른 채널 메시지가 섞입니다.** 채널 구분 없이 시간순으로 직전 메시지를 가져옵니다.
4. **긴 예제는 정답이 잘립니다.** 512 토큰을 넘으면 끝부분, 즉 정답부터 잘려 나갑니다.
5. **TTS 예제 약 1,858개는 입력이 전부 같은 문장**이라, 입력을 무시하는 습관이 생깁니다.

개선 방향: 추론 입력을 학습 형식에 맞추기, assistant 응답에만 loss 적용, 채널별로 맥락 구성, 맥락을 앞쪽부터 자르기. 필요하면 Base 모델(`Qwen/Qwen2.5-7B`)로 다시 학습합니다.

## 예전 RAG 버전

`scripts/filter_messages.py`와 `scripts/embed_and_store.py`는 파인튜닝 이전 버전(ChromaDB 검색 + Groq API)에서 쓰던 스크립트입니다. 이 버전의 봇 코드는 삭제돼서 지금은 참고용으로만 남아 있습니다.

## 프로젝트 구조

```
src/        디스코드 봇 (Node.js)
scripts/    데이터 준비, 학습, 추론 서버 (Python)
data/       채팅 원본과 학습 데이터 (레포에 미포함)
models/     LoRA 어댑터 (레포에 미포함)
logs/       대화 로그 (레포에 미포함)
```
