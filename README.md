# deepseek-bridge

STT (OpenAI Whisper) + DeepSeek chat proxy for G2 Playground AI Chat page.

## Endpoint

`POST /chat` body `{ audio_base64: str, sample_rate?: int=16000 }` → `{ transcript, reply }`

PCM (16 kHz mono int16 LE) → WAV → Whisper → DeepSeek `deepseek-chat` → ≤300-char reply.

## Deploy

Coolify app, env: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, optional `ALLOWED_ORIGINS`.

FQDN: `https://deepseek-bridge.coscore.us` (CF tunnel `proxmox-tunnel`).
