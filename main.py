"""
deepseek-bridge — STT (OpenAI Whisper) + DeepSeek chat for G2 Playground.

Endpoint: POST /chat
  body: { audio_base64: str }  # 16kHz mono int16 LE PCM, raw bytes base64-encoded
  returns: { transcript: str, reply: str }

Wraps PCM in WAV header before Whisper, sends transcript to DeepSeek chat,
returns short HUD-friendly reply (~300 chars max, no markdown).
"""

import base64
import logging
import os
import struct
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ds-bridge")

GROQ_KEY = os.environ["GROQ_API_KEY"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3-turbo")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = (
    "You are an AI assistant displayed on Even G2 smart glasses (576x288 px, "
    "4-bit greyscale, ~400 chars per screen). Reply concisely in plain text, "
    "max 300 characters, no markdown, no code blocks. Match the user's language."
)

app = FastAPI(title="DeepSeek Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    audio_base64: str
    sample_rate: Optional[int] = 16000


class ChatResponse(BaseModel):
    transcript: str
    reply: str


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    data_size = len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_size)
        + pcm
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        pcm = base64.b64decode(req.audio_base64)
    except Exception as e:
        raise HTTPException(400, f"invalid base64: {e}")
    if len(pcm) < 3200:  # < 0.1s at 16kHz mono 16-bit
        raise HTTPException(400, "audio too short (<0.1s)")

    wav = pcm_to_wav(pcm, sample_rate=req.sample_rate or 16000)
    log.info("chat req: pcm=%dB wav=%dB sr=%d", len(pcm), len(wav), req.sample_rate)

    async with httpx.AsyncClient(timeout=30) as client:
        # STT — Groq Whisper (OpenAI-compatible)
        stt_r = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": STT_MODEL},
        )
        if stt_r.status_code != 200:
            log.error("stt fail %s: %s", stt_r.status_code, stt_r.text[:200])
            raise HTTPException(502, f"stt failed: {stt_r.status_code}")
        transcript = (stt_r.json().get("text") or "").strip()
        log.info("transcript: %s", transcript)
        if not transcript:
            return ChatResponse(transcript="", reply="(тишина — ничего не распознал)")

        # Chat — DeepSeek
        chat_r = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                "max_tokens": 200,
                "temperature": 0.4,
            },
        )
        if chat_r.status_code != 200:
            log.error("chat fail %s: %s", chat_r.status_code, chat_r.text[:200])
            raise HTTPException(502, f"deepseek failed: {chat_r.status_code}")
        reply = (chat_r.json()["choices"][0]["message"]["content"] or "").strip()
        if len(reply) > 300:
            reply = reply[:297] + "..."
        log.info("reply: %s", reply)

    return ChatResponse(transcript=transcript, reply=reply)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "deepseek-bridge"}
