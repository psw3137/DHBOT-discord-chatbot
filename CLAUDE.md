# CLAUDE.md — DHBOT 프로젝트 작업 로그 및 인수인계

> 이 파일은 모든 작업 내역, 결정 사항, 다음 대화를 위한 컨텍스트를 기록합니다.

---

## 프로젝트 개요

**프로젝트명**: Discord Mimic Bot (DHBOT)
**목적**: 친구의 디스코드 채팅 데이터(2년치)를 학습해 말투를 흉내내는 봇
**모방 대상**: 닉네임은 `.env`의 `FRIEND_NAME`으로 지정

---

## 현재 상태 (2026-03-26 기준)

### 완성된 것

- discord.js 봇 기본 동작 (멘션 감지, 응답, 대화 맥락 유지)
- RAG 파이프라인: ChromaDB + sentence-transformers + Python 쿼리 서버
- Groq API 연동 (llama-3.3-70b-versatile)
- 데이터 파이프라인: 4개 JSON → 대화쌍 추출 → 임베딩 저장
- 디스코드 서버 내 봇 정상 작동 확인

### QLoRA 파인튜닝 환경 구축 완료 (2026-03-25)

- PyTorch 2.4.1+cu124 (CUDA 12.4) 설치 완료 → `torch.cuda.is_available() = True`
- GPU: RTX 3080 10GB 확인
- 패키지: transformers, peft, trl, bitsandbytes, datasets, accelerate 설치 완료

### 파인튜닝 데이터 준비 완료 (2026-03-25)

- `scripts/prepare_finetune.py` 작성 완료
- **데이터 전략**: 전체 대화 맥락 포함 + TTS 하이브리드
  - raw/ JSON 4개 전체 사용 (190,185개 메시지, 90,678개 턴으로 그룹핑)
  - 일반방/가챠겜방/근버스방: 직전 15개 메시지를 컨텍스트로 → 맥락-응답 쌍 11,378개
  - TTS방: 음성→TTS 변환이라 맥락 신뢰 불가 → 대상 친구 발화 자체만 스타일 학습 데이터로 1,858개
  - 연속 발화(5분 이내 같은 채널) 그룹핑, 시간 갭 3시간 초과 시 제외
- 결과: **13,236개 학습 예제** → `data/finetune_data.jsonl`

### 파인튜닝 완료 (2026-03-26)

- 베이스 모델: `Qwen/Qwen2.5-7B-Instruct` (EXAONE 3.5/3.0 → transformers 호환 문제로 변경)
- Python 3.14 + PyTorch 2.11.0+cu128 환경에서 실행
- QLoRA (4-bit nf4, LoRA r=16), max_length 512, 실질 배치 8, 에폭 3, LR 2e-4
- 총 4965 스텝, 5시간 19분 (3.86s/it)
- 중간에 CUDA unknown error로 중단 → 체크포인트에서 재개 후 완료
- **LoRA 어댑터 저장 완료**: `models/friend-lora/`

### 봇 연결 완료 (2026-03-26)

- `scripts/merge_lora.py` — LoRA 어댑터 머지 완료 → `models/friend-merged/`
- `scripts/inference_server.py` — 파인튜닝 모델 추론 서버 (포트 8002)
  - Qwen2.5-7B 4-bit + LoRA 어댑터 로드, HTTP POST 요청 처리
- `src/local_model.js` — 추론 서버 호출 (Node.js 내장 fetch 사용)
- `src/handler.js` — Groq → callLocalModel 교체 완료

### 실행 방법 (파인튜닝 버전)

```powershell
# 터미널 1: 추론 서버 (먼저 실행, 로드 1~2분)
py -3.14 scripts/inference_server.py

# 터미널 2: 봇
node src/index.js
```

### 코드 정리 완료 (2026-03-26)

- 미사용 파일 삭제: `src/chroma.js`, `src/prompt.js`, `src/groq.js`, `scripts/query_server.py`
- `src/handler.js` — 맥락을 채널 단위 → **유저 단위**(`message.author.id`)로 변경
- `src/handler.js` — `@봇 !초기화` 명령 추가 (본인 맥락 리셋)

### 현재 설정값 (2026-03-26)

- `CONTEXT_LENGTH=0` — 맥락 완전 비활성화 (다중 유저 오염 방지)
- 시스템 프롬프트 없음 — 학습 때와 동일한 조건
- `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.15`

### 알려진 문제 및 개선 과제

- **근본 한계**: `Qwen2.5-7B-Instruct` (RLHF 정렬) + LoRA 0.5% 업데이트 조합으로는
  학습 분포 밖 입력(지시형 문장, 일반 질문 등)에서 학습 데이터 파편을 랜덤 생성하는 현상 발생.
  학습 패턴(짧은 욕설/친구 비하/가챠 관련)은 어느 정도 동작하나 범용성 없음.
- **근본 해결**: `Qwen/Qwen2.5-7B` **(Base 모델)으로 재학습** 필요.
  재학습 시 `finetune.py`에서 `BASE_MODEL = 'Qwen/Qwen2.5-7B'`로 변경하고
  `trainer.train(resume_from_checkpoint=False)`으로 처음부터 실행.
- **추가 학습 데이터**: `logs/chat.log`에 쌓인 대화 중 좋은 응답만 수동으로 골라 `finetune_data.jsonl`에 추가 가능.
  단, 현재 로그에 이상한 응답이 다수 포함되어 있어 오염 주의.

### 실행 방법 (RAG 버전, 파인튜닝 서버 없을 때)

```powershell
# 터미널 1
python scripts/query_server.py
# 터미널 2
node src/index.js   # handler.js를 Groq 버전으로 되돌려야 함
```

---

## 데스크톱 이전 및 환경 구축 가이드

### STEP 0. 코드 옮기기

현재 노트북의 `c:\project\DHBOT` 폴더 전체를 데스크톱으로 복사.
아래 폴더는 제외하고 데스크톱에서 재생성:

- `node_modules/` → `npm install`로 재설치
- `chroma_db/` → `embed_and_store.py`로 재생성
- `.venv/` → pip으로 재설치

복사해야 할 핵심 파일:

```text
DHBOT/
├── src/                  ← 봇 소스코드
├── scripts/              ← Python 스크립트
├── data/
│   ├── raw/              ← JSON 원본 4개 (필수)
│   └── friend_messages.json
├── .env                  ← API 키 포함 (필수)
├── package.json
├── requirements.txt
└── CLAUDE.md
```

---

### STEP 1. 기본 개발 환경

**Node.js 20+**

- <https://nodejs.org> → LTS 버전 다운로드 및 설치

**Python 3.10+**

- <https://python.org/downloads> → 3.10 이상
- 설치 시 **"Add Python to PATH"** 반드시 체크

설치 확인:

```bash
node -v
npm -v
python --version
```

---

### STEP 2. CUDA 환경 (파인튜닝용)

RTX 3080은 CUDA 12.x 지원.

1. NVIDIA 드라이버 최신 버전 설치: <https://www.nvidia.com/drivers>
2. CUDA Toolkit 12.1 설치: <https://developer.nvidia.com/cuda-12-1-0-download-archive>

설치 확인:

```bash
nvidia-smi
nvcc --version
```

---

### STEP 3. Node.js 패키지 설치

```bash
cd DHBOT
npm install
```

설치되는 패키지: `discord.js`, `groq-sdk`, `dotenv`, `node-fetch`

---

### STEP 4. Python 패키지 설치 (봇 운영용)

```bash
pip install -r requirements.txt
```

`requirements.txt` 내용: `chromadb`, `sentence-transformers`, `python-dotenv`

---

### STEP 5. Python 패키지 설치 (파인튜닝용)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes datasets accelerate
```

설치 확인:

```python
import torch
print(torch.cuda.is_available())      # True 여야 함
print(torch.cuda.get_device_name(0))  # RTX 3080 출력 여야 함
```

---

### STEP 6. .env 파일 확인

복사된 `.env` 파일에 아래 값이 채워져 있어야 함:

```env
DISCORD_TOKEN=...
GROQ_API_KEY=...
FRIEND_NAME=친구의_디스코드_닉네임
CONTEXT_LENGTH=10
CHROMA_HOST=http://localhost:8000
RAG_TOP_K=30
GROQ_MODEL=llama-3.3-70b-versatile
```

---

### STEP 7. 데이터 파이프라인 재실행

`chroma_db/`를 새로 만들어야 함:

```bash
python scripts/filter_messages.py
python scripts/embed_and_store.py
```

---

### STEP 8. 봇 실행 (현재 RAG 방식)

```bash
# 터미널 1: 쿼리 서버
python scripts/query_server.py

# 터미널 2: 봇
node src/index.js
```

---

## 다음 작업: QLoRA 파인튜닝

환경 구축 완료 후 새 대화에서 진행할 작업.

### 목표

2년치 채팅 데이터로 LLM 파인튜닝 → Groq API 대신 로컬 모델 사용

### 계획

1. **학습 데이터 변환** (`scripts/prepare_finetune.py` 작성 예정)
   - `data/friend_messages.json` (대화쌍) → JSONL 포맷으로 변환

   ```json
   {"messages": [
     {"role": "user", "content": "이전 대화 맥락"},
     {"role": "assistant", "content": "대상 친구 응답"}
   ]}
   ```

2. **베이스 모델**: EXAONE 3.5 7.8B (LG AI, 한국어 특화)
   - HuggingFace: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`

3. **파인튜닝** (`scripts/finetune.py` 작성 예정)
   - 방법: QLoRA (4-bit 양자화 + LoRA)
   - 예상 시간: RTX 3080 기준 6~10시간

4. **추론 서버**: Ollama로 파인튜닝된 모델 서빙

5. **봇 연결**: `src/groq.js` → Ollama API 호출로 교체

---

## 프로젝트 구조

```text
DHBOT/
├── src/
│   ├── index.js          봇 진입점, clientReady 이벤트
│   ├── handler.js        멘션 감지, 유저별 맥락(Map), 응답 전송, 로그, !초기화 명령
│   └── local_model.js    추론 서버(localhost:8002) 호출
├── scripts/
│   ├── inference_server.py   로컬 모델 추론 서버 (포트 8002)
│   ├── finetune.py           QLoRA 학습 스크립트
│   ├── merge_lora.py         LoRA 어댑터 머지
│   ├── prepare_finetune.py   학습 데이터 준비
│   ├── embed_and_store.py    (RAG용, 미사용)
│   └── filter_messages.py    (RAG용, 미사용)
├── data/
│   ├── raw/                  원본 JSON 4개 (일반방, TTS, 가챠겜방, 근버스방)
│   ├── friend_messages.json  대화쌍 21,506개
│   └── finetune_data.jsonl   학습 데이터 13,236개
├── models/
│   ├── friend-lora/        LoRA 어댑터
│   └── friend-merged/      머지된 단일 모델
├── logs/
│   └── chat.log              대화 로그 (JSON Lines)
├── .env                      API 키 (커밋 금지)
├── package.json
└── requirements.txt
```

---

## 캐릭터 설정 (prompt.js에 반영됨)

### 대상 친구 (모방 대상)

- 눈치가 약간 없고 근거없는 자신감이 넘침
- 욕설을 가끔 씀. "야이 시발년아"를 자주 쓰지만 매번은 아님
- 서브컬쳐, 가챠겜 좋아함. 특히 어린 여동생 캐릭터류 선호
- Discord 닉네임: `.env`의 `FRIEND_NAME`

### 친구 B

- 대상 친구의 가장 친한 친구. 평소엔 놀리지만 뒤에선 살짝 챙겨줌

### 용어

- 픽뚫: 가챠겜에서 원하는 캐릭터 대신 다른 캐릭터가 뽑히는 것. 나쁜 일.
- "픽뚫 축하해" = 놀리는 것, 진짜 축하 아님

---

## 주요 결정 사항

| 날짜 | 결정 | 이유 |
| --- | --- | --- |
| 2026-03-24 | data/raw/에 채널별 JSON 분리 보관 | 다중 파일 대응 |
| 2026-03-24 | filter_messages.py를 nickname 기준 매칭으로 변경 | author.name이 핸들이라 닉네임과 다름 |
| 2026-03-24 | ChromaDB HTTP 서버 대신 Python 쿼리 서버(8001) 사용 | ChromaDB 1.x API 변경으로 query_texts 미지원 |
| 2026-03-24 | 단발 발화 → 대화쌍(이전 2줄 + 응답)으로 변경 | RAG 샘플에 맥락 부여 |
| 2026-03-24 | frequency_penalty 1.0, stripForeignChars 후처리 | 외국어 토큰 생성 방지 |
| 2026-03-25 | Groq → QLoRA 파인튜닝으로 전환 예정 | RAG+프롬프트의 근본 한계 해결 |
| 2026-03-26 | CONTEXT_LENGTH=0 (맥락 비활성화) | 다중 유저 채널에서 맥락 오염으로 응답 품질 급격히 저하 |
| 2026-03-26 | 맥락 채널 단위 → 유저 단위로 변경 | 여러 유저가 같은 채널에서 질문 시 맥락 뒤섞임 |
| 2026-03-26 | 시스템 프롬프트 제거 | 학습 시 시스템 프롬프트 없었음. 추가 시 오히려 응답 품질 저하 확인 |
