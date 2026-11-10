// 봇 진입점 — 클라이언트 초기화 및 이벤트 등록
// NFR-02: 전역 에러 핸들러
require('dotenv').config();
const { Client, GatewayIntentBits } = require('discord.js');
const handler = require('./handler');

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

client.once('clientReady', () => {
  console.log(`봇 로그인 완료: ${client.user.tag}`);
});

client.on('messageCreate', handler);

process.on('uncaughtException', (err) => console.error('[uncaughtException]', err));
process.on('unhandledRejection', (err) => console.error('[unhandledRejection]', err));

client.login(process.env.DISCORD_TOKEN);
