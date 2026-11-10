// 파인튜닝 모델 추론 서버 (localhost:8002) 호출
const INFERENCE_URL = 'http://localhost:8002';

async function callLocalModel(context, userMessage) {
  // context(이전 대화) + 현재 메시지를 messages 배열로 조합
  const messages = [
    ...context,
    { role: 'user', content: userMessage },
  ];

  const res = await fetch(INFERENCE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, max_new_tokens: 200 }),
  });

  if (!res.ok) throw new Error(`추론 서버 오류: ${res.status}`);

  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.response;
}

module.exports = { callLocalModel };
