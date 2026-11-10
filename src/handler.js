// FR-01: 멘션 감지 / FR-04: 대화 맥락 유지 / FR-06: 응답 전송
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { callLocalModel } = require('./local_model');

const LOG_PATH = path.join(__dirname, '../logs/chat.log');
fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });

function writeLog(entry) {
  const line = JSON.stringify({ ts: new Date().toISOString(), ...entry }) + '\n';
  fs.appendFileSync(LOG_PATH, line, 'utf-8');
  console.log(line.trim());
}

const CONTEXT_LENGTH = parseInt(process.env.CONTEXT_LENGTH || '10', 10);
const MAX_DISCORD_LENGTH = 2000;

// 유저 단위 대화 맥락 (Map<userId, messages[]>)
const contextMap = new Map();

function getContext(userId) {
  if (!contextMap.has(userId)) contextMap.set(userId, []);
  return contextMap.get(userId);
}

function addToContext(userId, role, content) {
  const ctx = getContext(userId);
  ctx.push({ role, content });
  // CONTEXT_LENGTH 초과 시 오래된 항목 제거 (FIFO)
  while (ctx.length > CONTEXT_LENGTH) ctx.shift();
}

function truncate(text) {
  if (text.length <= MAX_DISCORD_LENGTH) return text;
  return text.slice(0, 1997) + '...';
}

module.exports = async function handler(message) {
  // 봇 자신의 메시지 무시
  if (message.author.bot) return;
  // 멘션된 경우에만 반응
  if (!message.mentions.has(message.client.user)) return;

  // <@봇ID> 제거 후 순수 텍스트 추출
  const userText = message.content
    .replace(/<@!?\d+>/g, '')
    .trim();

  // 맥락 초기화 명령
  if (userText === '!초기화') {
    contextMap.set(message.author.id, []);
    await message.reply('맥락 초기화했어');
    return;
  }

  // 멘션만 하고 내용 없는 경우
  const queryText = userText || '안녕';

  try {
    await message.channel.sendTyping();

    const context = getContext(message.author.id);
    const reply = await callLocalModel(context, queryText);

    addToContext(message.author.id, 'user', queryText);
    addToContext(message.author.id, 'assistant', reply);

    writeLog({ user: message.author.username, input: queryText, reply });
    await message.reply(truncate(reply));
  } catch (err) {
    console.error('[handler] 오류:', err);

    if (err.status === 429) {
      await message.reply('잠깐만, 너무 많이 불렀잖아 ㅋㅋ');
    } else {
      await message.reply('오류 발생했어, 나중에 다시 해봐');
    }
  }
};
