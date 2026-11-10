"""
파인튜닝 데이터 준비 스크립트 (전체 대화 맥락 버전)
- data/raw/*.json → data/finetune_data.jsonl

채널별 처리 전략:
  일반방/가챠겜방/근버스방: 대상 친구 발화 직전 대화 맥락을 컨텍스트로 사용 (맥락-응답 쌍)
  TTS방: 음성채팅 TTS라 맥락 신뢰 불가 → 대상 친구 발화만 스타일 학습 데이터로 사용
"""

import json
import os
import glob
import re
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), '../data/raw')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/finetune_data.jsonl')

FRIEND_NICKNAME = os.environ.get('FRIEND_NAME', '')  # .env에서 대상 친구 닉네임 지정
CONTEXT_WINDOW = 15          # 발화 직전 최대 N개 메시지를 컨텍스트로
BURST_GAP_MINUTES = 5        # N분 이내 연속 발화는 하나로 묶음
MAX_CONTEXT_GAP_HOURS = 3    # 컨텍스트와 발화 사이 갭이 이 이상이면 제외
TTS_STYLE_PROMPT = '자연스럽게 한마디 해봐.'  # TTS 스타일 학습용 고정 user 프롬프트
TTS_MIN_LENGTH = 2           # TTS 발화 최소 길이 (너무 짧은 단편 제외)

SYSTEM_PROMPT = (
    f"너는 {FRIEND_NICKNAME}야. 디스코드 서버에서 친구들과 대화하는 상황이야. "
    "눈치가 약간 없고 근거없는 자신감이 넘치며, 가끔 욕설을 쓰고, "
    "서브컬쳐와 가챠겜을 좋아해. 아래 대화 흐름을 보고 자연스럽게 이어서 말해."
)


def parse_timestamp(ts_str):
    """
    Python 3.9 호환 타임스탬프 파싱.
    '2024-01-01T17:19:01.27+09:00' 같이 소수점 자릿수가 불규칙한 경우 처리.
    """
    # 소수점 부분을 6자리로 정규화
    ts_str = re.sub(r'\.(\d+)([+-])', lambda m: f'.{m.group(1).ljust(6, "0")}{m.group(2)}', ts_str)
    return datetime.fromisoformat(ts_str)


def load_all_messages():
    """모든 raw JSON에서 메시지 로드 후 시간순 정렬"""
    all_msgs = []
    for path in glob.glob(os.path.join(RAW_DIR, '*.json')):
        channel_name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for m in data.get('messages', []):
            content = m.get('content', '').strip()
            if not content:
                continue
            if m['author'].get('isBot', False):
                continue
            all_msgs.append({
                'ts': parse_timestamp(m['timestamp']),
                'nickname': m['author']['nickname'],
                'content': content,
                'channel': channel_name,
            })

    all_msgs.sort(key=lambda x: x['ts'])
    return all_msgs


def group_bursts(msgs):
    """같은 사람이 BURST_GAP_MINUTES 이내에 연속으로 보낸 메시지를 하나로 묶음."""
    if not msgs:
        return []

    grouped = []
    cur = dict(msgs[0])
    burst_limit = timedelta(minutes=BURST_GAP_MINUTES)

    for m in msgs[1:]:
        same_person = (m['nickname'] == cur['nickname'])
        close_in_time = (m['ts'] - cur['ts']) < burst_limit
        same_channel = (m['channel'] == cur['channel'])
        if same_person and close_in_time and same_channel:
            cur['content'] += '\n' + m['content']
            cur['ts'] = m['ts']
        else:
            grouped.append(cur)
            cur = dict(m)
    grouped.append(cur)
    return grouped


def build_training_examples(msgs):
    examples = []
    skipped_no_ctx = 0
    skipped_tts_short = 0
    max_gap = timedelta(hours=MAX_CONTEXT_GAP_HOURS)

    for i, m in enumerate(msgs):
        if m['nickname'] != FRIEND_NICKNAME:
            continue

        # TTS방: 맥락 무시, 발화 자체만 스타일 학습 데이터로 사용
        if m['channel'] == 'TTS':
            if len(m['content'].strip()) < TTS_MIN_LENGTH:
                skipped_tts_short += 1
                continue
            examples.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': TTS_STYLE_PROMPT},
                    {'role': 'assistant', 'content': m['content']},
                ]
            })
            continue

        # 일반방/가챠겜방/근버스방: 직전 대화 맥락 수집
        context_msgs = []
        for j in range(i - 1, max(i - CONTEXT_WINDOW - 1, -1), -1):
            ctx = msgs[j]
            if (m['ts'] - ctx['ts']) > max_gap:
                break
            context_msgs.insert(0, ctx)

        if not context_msgs:
            skipped_no_ctx += 1
            continue

        context_lines = [
            f"[{c['nickname']}]: {c['content']}" for c in context_msgs
        ]
        context_text = '\n'.join(context_lines)

        examples.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': context_text},
                {'role': 'assistant', 'content': m['content']},
            ]
        })

    return examples, skipped_no_ctx, skipped_tts_short


def main():
    print('메시지 로드 중...')
    all_msgs = load_all_messages()
    print(f'전체 메시지: {len(all_msgs)}개')

    print('연속 발화 그룹핑 중...')
    grouped = group_bursts(all_msgs)
    print(f'그룹핑 후: {len(grouped)}개 턴')

    dh_count = sum(1 for m in grouped if m['nickname'] == FRIEND_NICKNAME)
    print(f'대상 친구 턴: {dh_count}개')

    print('학습 데이터 생성 중...')
    examples, skipped_no_ctx, skipped_tts_short = build_training_examples(grouped)

    tts_style = sum(1 for e in examples if e['messages'][1]['content'] == TTS_STYLE_PROMPT)
    ctx_response = len(examples) - tts_style

    print(f'\n결과:')
    print(f'  학습 예제 합계: {len(examples)}개')
    print(f'    - 맥락-응답 쌍 (일반방/가챠겜방/근버스방): {ctx_response}개')
    print(f'    - 스타일 학습 (TTS방): {tts_style}개')
    print(f'  제외 (컨텍스트 없음/시간초과): {skipped_no_ctx}개')
    print(f'  제외 (TTS 발화 너무 짧음): {skipped_tts_short}개')

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f'\n저장: {OUTPUT_PATH}')

    # 샘플 출력
    print('\n--- 샘플 ---')
    sample = examples[10]
    print('[user context]')
    print(sample['messages'][1]['content'])
    print('[assistant]')
    print(sample['messages'][2]['content'])


if __name__ == '__main__':
    main()
