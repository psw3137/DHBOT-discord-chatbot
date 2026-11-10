"""
FR-07: DiscordChatExporter JSON → 대화 쌍 추출
- 대상 친구의 발화 직전 2개 메시지 + 친구 응답을 하나의 쌍으로 저장
입력: data/raw/*.json
출력: data/friend_messages.json
실행: python scripts/filter_messages.py
"""

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

FRIEND_NAME = os.environ.get("FRIEND_NAME", "")
RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/friend_messages.json")

CONTEXT_WINDOW = 2        # 친구 응답 직전 메시지 수
MAX_GAP_MINUTES = 10      # 직전 메시지와 시간 간격이 이 이상이면 맥락 없는 것으로 간주

def is_valid_content(content: str) -> bool:
    if not content or not content.strip():
        return False
    # URL만 있는 메시지 제외
    stripped = re.sub(r"https?://\S+", "", content).strip()
    if not stripped:
        return False
    # 멘션/이모지 태그만 있는 메시지 제외
    stripped = re.sub(r"<[^>]+>", "", stripped).strip()
    if not stripped:
        return False
    return True

def parse_timestamp(ts: str) -> float:
    from datetime import datetime, timezone
    # "2024-01-01T00:01:07.881+09:00" 형태 파싱
    try:
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except Exception:
        return 0.0

def extract_pairs_from_file(filepath: str) -> list[str]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    pairs = []

    for i, msg in enumerate(messages):
        nickname = msg.get("author", {}).get("nickname", "")
        content = msg.get("content", "")

        if nickname != FRIEND_NAME:
            continue
        if not is_valid_content(content):
            continue

        # 직전 메시지들 수집 (다른 사람 발화, 시간 간격 이내)
        friend_ts = parse_timestamp(msg.get("timestamp", ""))
        context_lines = []

        for j in range(i - 1, max(i - 10, -1), -1):
            prev = messages[j]
            prev_nick = prev.get("author", {}).get("nickname", "")
            prev_content = prev.get("content", "")
            prev_ts = parse_timestamp(prev.get("timestamp", ""))

            # 시간 간격 초과 시 중단
            if friend_ts - prev_ts > MAX_GAP_MINUTES * 60:
                break
            # 봇 메시지 제외
            if prev.get("author", {}).get("isBot", False):
                continue
            if not is_valid_content(prev_content):
                continue

            context_lines.insert(0, f"{prev_nick}: {prev_content.strip()}")
            if len(context_lines) >= CONTEXT_WINDOW:
                break

        # 직전 대화가 없으면 단발 발화로만 저장
        if context_lines:
            pair = "\n".join(context_lines) + f"\n{FRIEND_NAME}: {content.strip()}"
        else:
            pair = f"{FRIEND_NAME}: {content.strip()}"

        pairs.append(pair)

    return pairs

def main():
    if not FRIEND_NAME:
        print("오류: .env 파일에 FRIEND_NAME이 설정되지 않았습니다.")
        return

    json_files = sorted([
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".json")
    ])

    if not json_files:
        print(f"오류: {RAW_DIR} 폴더에 JSON 파일이 없습니다.")
        return

    all_pairs = []
    for filepath in json_files:
        filename = os.path.basename(filepath)
        pairs = extract_pairs_from_file(filepath)
        print(f"  {filename}: {len(pairs)}개 대화쌍 추출")
        all_pairs.extend(pairs)

    # 중복 제거
    seen = set()
    unique = []
    for p in all_pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"\n완료: 총 {len(unique)}개 대화쌍 → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
