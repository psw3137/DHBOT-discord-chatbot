"""
FR-08: 임베딩 생성 및 ChromaDB 저장
입력: data/friend_messages.json
출력: chroma_db/ (로컬 벡터DB)
실행: python scripts/embed_and_store.py
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

import chromadb
from sentence_transformers import SentenceTransformer

INPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/friend_messages.json')
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '../chroma_db')
COLLECTION_NAME = 'friend_messages'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 500

def main():
    with open(INPUT_FILE, encoding='utf-8') as f:
        messages = json.load(f)
    total = len(messages)
    print(f'총 {total}개 발화 로드 완료')

    print(f'임베딩 모델 로드 중: {MODEL_NAME}')
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 기존 컬렉션 초기화
    try:
        client.delete_collection(COLLECTION_NAME)
        print('기존 컬렉션 초기화 완료')
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    for i in range(0, total, BATCH_SIZE):
        batch = messages[i:i + BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False).tolist()
        ids = [str(i + j) for j in range(len(batch))]
        collection.add(documents=batch, embeddings=embeddings, ids=ids)
        print(f'저장 중... {min(i + BATCH_SIZE, total)}/{total}')

    print(f'\n완료: {total}개 발화 → {CHROMA_PATH}')

if __name__ == '__main__':
    main()
