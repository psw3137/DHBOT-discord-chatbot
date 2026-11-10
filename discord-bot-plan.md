# 디스코드 말투 모방 봇 — 프로젝트 계획서 (RAG 버전)

## 1. 프로젝트 개요

친구의 디스코드 채팅 데이터(최근 2~3년치)를 기반으로, 해당 친구의 말투를 흉내내는 디스코드 봇을 제작한다.
사용자가 봇을 @멘션하면 대화 내용과 의미적으로 유사한 친구 발화를 벡터 검색으로 꺼내와
Groq API에 전달함으로써 맥락에 맞는 말투 재현 품질을 극대화한다.

- **목적**: 친구들끼리 재미로 사용하는 가벼운 봇
- **트리거**: 봇 @멘션
- **LLM**: Groq API (무료 티어)
- **RAG**: ChromaDB (로컬 벡터DB) + 임베딩 모델
- **실행 환경**: 로컬 PC (사용할 때만 실행)
- **모방 대상**: 1명
- **학습 데이터 범위**: 최근 2~3년치 채팅

---

## 2. 기술 스택

| 분류 | 기술 |
|------|------|
| 런타임 | Node.js 20+ |
| 봇 라이브러리 | discord.js v14 |
| LLM API | Groq SDK (`groq-sdk`) |
| 벡터DB | ChromaDB (로컬, Python) |
| 임베딩 | `sentence-transformers` (Python, 무료 로컬 모델) |
| Node ↔ Python 연동 | ChromaDB HTTP 서버 (`chroma run`) |
| 환경변수 | dotenv |
| 데이터 추출 | DiscordChatExporter (CLI or GUI) |
| 데이터 가공 | Python 스크립트 |

> **왜 Python 병행?**
> ChromaDB와 임베딩 생성(`sentence-transformers`)이 Python 생태계에서 가장 안정적이다.
> 봇 자체(discord.js)는 Node.js로 유지하고, ChromaDB만 별도 로컬 서버로 실행한다.

---

## 3. 전체 아키텍처

```
[오프라인 1회 작업]
DiscordChatExporter
        ↓ JSON 추출 (날짜 범위 필터링)
filter_messages.py
        ↓ 친구 발화만 추출
embed_and_store.py
        ↓ 문장 임베딩 생성 + ChromaDB 저장
ChromaDB (로컬)

[봇 실행 시]
사용자 @멘션
        ↓
discord.js (Node.js)
        ↓ 멘션 메시지 텍스트
ChromaDB HTTP API
        ↓ 의미적으로 유사한 친구 발화 Top 30개 반환
system prompt 생성 (샘플 30개 삽입)
        ↓
Groq API 호출
        ↓
친구 말투로 답장
```

---

## 4. 사전 준비 — 채팅 데이터 추출

### 4-1. DiscordChatExporter 설치
- GitHub: https://github.com/Tyrrrz/DiscordChatExporter
- GUI 버전(`DiscordChatExporter.zip`) 또는 CLI 버전 다운로드
- Windows 기준 압축 해제 후 바로 실행 가능

### 4-2. 디스코드 유저 토큰 획득
1. 디스코드 웹(discord.com/app) 접속
2. F12 → Network 탭 열기
3. 아무 채널 클릭 후 `messages` 요청 선택
4. Request Headers의 `authorization` 값 복사

> ⚠️ 유저 토큰은 절대 외부 공유 금지. .env 파일에만 보관.

### 4-3. 채널 ID 확인
- 디스코드 설정 → 고급 → 개발자 모드 활성화
- 원하는 채널 우클릭 → "ID 복사"

### 4-4. 날짜 범위 지정 추출
```bash
# 예시: 2022년 1월 1일 ~ 2025년 1월 1일 범위만 추출
DiscordChatExporter.Cli export \
  -t "유저토큰" \
  -c 채널ID \
  -f Json \
  --after 2022-01-01 \
  --before 2025-01-01 \
  -o ./data/
```
`--after`, `--before` 날짜는 원하는 범위로 자유롭게 조정한다.

---

## 5. 오프라인 데이터 파이프라인 (1회 실행)

### 5-1. 친구 발화 필터링 (`scripts/filter_messages.py`)
- 입력: `data/raw_export.json`
- 출력: `data/friend_messages.json`
- 로직: `author.name == FRIEND_NAME` 인 메시지의 `content`만 추출

```json
["ㅋㅋㅋ 그게 말이 되냐고", "아 진짜 왜 그랬어", "어이없어 진짜", ...]
```

### 5-2. 임베딩 생성 및 ChromaDB 저장 (`scripts/embed_and_store.py`)
- `sentence-transformers`의 다국어 모델 `paraphrase-multilingual-MiniLM-L12-v2` 사용
- 친구 발화 전체를 벡터로 변환 후 ChromaDB 컬렉션에 저장
- **1회만 실행**하면 되고, 이후 봇 실행 시마다 재실행 불필요

```bash
pip install chromadb sentence-transformers
python scripts/embed_and_store.py
```

---

## 6. 기능 명세

### 핵심 기능
| 기능 | 설명 |
|------|------|
| @멘션 감지 | 봇이 멘션되면 해당 메시지를 파이프라인으로 전달 |
| 벡터 유사도 검색 | ChromaDB에서 멘션 내용과 유사한 친구 발화 Top 30개 추출 |
| 말투 모방 답장 | 추출된 샘플 + system prompt → Groq API → 친구 말투 생성 |
| 대화 맥락 유지 | 최근 N개 메시지를 기억해 멀티턴 대화 지원 |

### 제외 기능 (MVP 범위 밖)
- 슬래시 커맨드
- 봇 on/off 토글
- 말투 강도 조절

---

## 7. 프롬프트 설계

### System Prompt 구조
```
너는 [친구 이름]이야.
아래는 지금 대화 주제와 관련된 [친구 이름]의 실제 디스코드 채팅 예시야.
이 사람의 말투, 어휘 선택, 이모지/이모티콘 사용 방식, 문장 길이, 반응 패턴을 최대한 흉내내서 대답해줘.
절대 표준어나 공손한 말투를 쓰지 마. 예시에서 보이는 그대로의 어투를 유지해.
일부 예시는 앞 맥락 없이 [친구 이름]의 발화만 있을 수 있어. 그건 말투 스타일 참고용으로만 활용해.

--- 관련 채팅 예시 시작 ---
[ChromaDB 벡터 검색 결과 Top 30개 동적 삽입]
--- 관련 채팅 예시 끝 ---
```

> **RAG 방식의 장점**: 스포츠 얘기엔 스포츠 관련 발화, 게임 얘기엔 게임 관련 발화가 자동으로 선택되어
> 맥락에 맞는 말투 재현 품질이 고정 샘플 방식보다 높아진다.

---

## 8. 폴더 구조

```
discord-bot/
├── src/
│   ├── index.js             # 봇 진입점, 클라이언트 초기화
│   ├── handler.js           # messageCreate 이벤트 처리
│   ├── groq.js              # Groq API 호출 로직
│   ├── prompt.js            # system prompt 생성 함수
│   └── chroma.js            # ChromaDB HTTP API 호출 (유사 샘플 검색)
├── scripts/
│   ├── filter_messages.py   # JSON → 친구 발화만 추출 (1회)
│   └── embed_and_store.py   # 임베딩 생성 + ChromaDB 저장 (1회)
├── data/
│   ├── raw_export.json      # DiscordChatExporter 원본 출력
│   └── friend_messages.json # 가공된 친구 발화 목록
├── chroma_db/               # ChromaDB 로컬 저장소 (자동 생성)
├── .env
├── .gitignore
└── package.json
```

---

## 9. 환경변수 (.env)

```env
DISCORD_TOKEN=디스코드_봇_토큰
GROQ_API_KEY=그록_API_키
FRIEND_NAME=친구_닉네임
CONTEXT_LENGTH=10            # 대화 맥락으로 기억할 메시지 수
CHROMA_HOST=http://localhost:8000
RAG_TOP_K=30                 # 벡터 검색으로 가져올 샘플 수
```

---

## 10. 핵심 로직 흐름

```
사용자가 @봇 멘션
        ↓
messageCreate 이벤트 감지
        ↓
ChromaDB에 멘션 텍스트로 유사도 검색 (Top 30)
        ↓
system prompt 생성 (검색된 샘플 삽입)
        ↓
최근 대화 맥락 + 현재 메시지 → Groq API 전송
        ↓
응답 텍스트 → 디스코드 채널에 답장
```

---

## 11. 개발 단계 (Phase)

### Phase 1 — 환경 세팅
- [ ] Node.js 프로젝트 초기화 (`npm init`)
- [ ] Node 패키지 설치 (`discord.js`, `groq-sdk`, `dotenv`)
- [ ] Python 환경 준비 (`pip install chromadb sentence-transformers`)
- [ ] 디스코드 Developer Portal에서 봇 생성 및 토큰 발급
- [ ] Groq 계정 생성 및 API 키 발급 (groq.com)
- [ ] .env 파일 구성

### Phase 2 — 데이터 파이프라인 (1회)
- [ ] DiscordChatExporter로 날짜 범위 지정 추출 (`--after`, `--before`)
- [ ] `scripts/filter_messages.py` 작성 및 실행 → `friend_messages.json` 생성
- [ ] `scripts/embed_and_store.py` 작성 및 실행 → ChromaDB 저장 완료
- [ ] ChromaDB 검색 테스트 (쿼리 입력 → 유사 발화 반환 확인)

### Phase 3 — 봇 핵심 기능 구현
- [ ] `src/index.js` — 봇 로그인 및 이벤트 등록
- [ ] `src/chroma.js` — ChromaDB HTTP API 유사도 검색 함수
- [ ] `src/prompt.js` — system prompt 생성 함수 (RAG 샘플 삽입)
- [ ] `src/groq.js` — Groq API 호출 함수
- [ ] `src/handler.js` — 멘션 감지 → 전체 파이프라인 연결

### Phase 4 — 테스트 및 튜닝
- [ ] ChromaDB 서버 실행 (`chroma run --path ./chroma_db`)
- [ ] 봇 로컬 실행 후 멘션 테스트
- [ ] 말투 재현 품질 확인
- [ ] `RAG_TOP_K`, system prompt 지시문 조정

---

## 12. 실행 방법 (개발 완료 후)

```bash
# 터미널 1 — ChromaDB 서버 실행
chroma run --path ./chroma_db

# 터미널 2 — 디스코드 봇 실행
node src/index.js
```

---

## 13. 주요 패키지 설치 명령어

```bash
# Node.js
mkdir discord-bot && cd discord-bot
npm init -y
npm install discord.js groq-sdk dotenv

# Python
pip install chromadb sentence-transformers
```

---

## 14. 참고 링크

- Discord Developer Portal: https://discord.com/developers/applications
- DiscordChatExporter: https://github.com/Tyrrrz/DiscordChatExporter
- Groq 콘솔: https://console.groq.com
- Groq 무료 모델 목록: https://console.groq.com/docs/models
- ChromaDB 공식 문서: https://docs.trychroma.com
- sentence-transformers: https://www.sbert.net
- discord.js 공식 문서: https://discord.js.org
