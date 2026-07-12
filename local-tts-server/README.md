# CurrenX TTS + Voice Cloning Backend

A FastAPI service that combines [Kokoro](https://github.com/hexgrad/kokoro) (text-to-speech)
with [OpenVoice V2](https://github.com/myshell-ai/OpenVoice) (zero-shot voice cloning) to turn
text into speech using either a fixed preset voice or a voice cloned from a short user-uploaded
audio sample.

## How it works

Two models, two jobs, chained together:

1. **Kokoro** generates the actual speech audio from text. It only knows a fixed bank of
   pre-trained voice presets (`af_heart`, `am_adam`, etc.) — it has no concept of an arbitrary
   uploaded voice.
2. **OpenVoice V2's Tone Color Converter** takes that generated speech and reshapes its timbre
   to match a reference audio clip, *after* generation. This is zero-shot — no training step,
   no per-user model — it extracts a speaker embedding from the reference clip at request time
   and applies it directly.

So a "cloned voice" request actually runs Kokoro twice-removed: Kokoro generates speech in a
neutral preset voice first, then OpenVoice reshapes that output to sound like the uploaded
sample. This is why `resolve_voice_embedding()` in `main.py` picks a plain preset for any
URL-shaped `voice` value — that's just the *base* pass, not the final output.

```
text + voice="af_heart"                 →  Kokoro  →  Kokoro's af_heart audio (done)
text + voice="<uploaded sample URL>"    →  Kokoro (neutral preset)  →  OpenVoice tone conversion  →  cloned-sounding audio
```

### Why cloning has a minimum text length

OpenVoice's speaker-embedding extractor needs a few seconds of audio to reliably compute an
embedding. Since Kokoro's *output* length is driven by *input* text length, very short text
produces audio too short to extract a usable embedding from. The backend enforces this with a
hard duration check (`MIN_CLONE_AUDIO_SECONDS` in `main.py`, currently 4.0s) and returns a
`400` with a clear message rather than silently falling back to the wrong voice. The frontend
mirrors this with a character-count nudge (`MIN_CLONE_TEXT_CHARS` in `App.jsx`) so users see
the warning before submitting.

### Why `/api/tts/stream` doesn't support cloning

Streaming yields audio chunk-by-chunk as Kokoro generates it. OpenVoice's tone conversion needs
the *complete* clip to extract a speaker embedding — there's no meaningful way to convert a
partial chunk. Clone requests sent to `/stream` fall back to the neutral base preset, logged
clearly server-side. Use `/api/tts/download` for anything involving a cloned voice.

## Project structure

```
local-tts-server/
├── main.py                  # FastAPI app — everything lives here currently
├── requirements.txt         # Linux-targeted pinned dependencies
├── Dockerfile
├── .dockerignore
├── checkpoints_v2/          # OpenVoice V2 weights (gitignored — see setup below)
│   └── converter/
│       ├── config.json
│       └── checkpoint.pth
└── processed/                # OpenVoice's own runtime cache (gitignored, safe to delete anytime)
```

## Local development setup

### macOS (Apple Silicon) or Linux

Should be close to friction-free — modern PyPI wheels exist for everything on these platforms.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS (Intel / x86_64)

**Read this before you `pip install`.** PyPI dropped Intel Mac wheel support for `llvmlite`
(used by `numba`, used by `librosa`) after a certain version, and PyTorch stopped shipping
Intel Mac wheels after `2.2.2`. Plain `pip install -r requirements.txt` **will fail** on this
platform trying to compile these from source, which typically doesn't succeed without a full
LLVM/cmake toolchain.

Use conda-forge instead, which still builds for Intel Mac:

```bash
# Install Miniforge if you don't have conda already
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh
# restart your terminal, then:

conda create -n tts-server python=3.11
conda activate tts-server
conda install -c conda-forge librosa=0.10.2   # pulls a working numba/llvmlite for this platform

pip install fastapi uvicorn python-dotenv kokoro soundfile pydub supabase requests
pip install torch==2.2.2 torchaudio==2.2.2     # last Intel Mac wheels PyTorch published
pip install git+https://github.com/myshell-ai/OpenVoice.git --no-deps
pip install wavmark faster-whisper inflect unidecode eng_to_ipa pypinyin cn2an jieba langid
pip install "transformers==4.46.3"             # newer transformers requires torch>=2.4, which Intel Mac can't get
conda install -c conda-forge "numpy<2"         # torch 2.2.2 predates NumPy 2.0's ABI change
```

This is why `requirements.txt` in this repo targets **Linux deployment specifically** and uses
newer, unrestricted versions — it is not meant to be installed as-is on an Intel Mac.

### Environment variables

Create a `.env` file one directory above `local-tts-server/` (i.e. at the `voice-clone-app`
root), containing:

```
VITE_SUPABASE_URL=your-project.supabase.co
VITE_SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

The service role key (not the anon key) is required — the backend writes to
`audio_generation_logs` storage and `tts_logs` table directly.

### Download the OpenVoice checkpoint

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='checkpoints_v2')"
```

~131MB, one-time. Cloning is disabled gracefully (presets still work) if this is missing —
check server startup logs for `OpenVoice converter loaded and ready` vs a `[WARNING]` line.

### Run it

```bash
python main.py
```

Server starts on `http://localhost:8000`.

## API reference

### `POST /api/tts/download`

Full-file generation. Supports both presets and cloning.

```json
{
  "text": "Text to synthesize (min ~80 chars if voice is a cloned sample URL)",
  "voice": "af_heart",
  "pitch_modifier": 1.0,
  "speed_modifier": 1.0
}
```

`voice` accepts:
- A preset ID: `af_heart`, `am_adam`, `af_bella`, `af_nicole`, `am_michael`, `bf_emma`, `bm_george`
- A blend: `"af_bella+af_nicole"` (any `+`-joined combination of presets, averaged)
- A fixed celebrity blend: `celebrity_jolie`, `celebrity_Mayor`
- An `http(s)://` URL to an uploaded audio sample — triggers OpenVoice cloning

Returns: `audio/wav` binary stream, or a JSON `{"detail": "..."}` error body on 4xx/5xx.

### `POST /api/tts/stream`

Chunked raw PCM streaming. Same `voice` field, but cloning is not applied (see above) —
clone-shaped `voice` values silently use the neutral base preset here.

## Deploying

### Why not Netlify / Cloudflare

Both are static-site / edge-function platforms. Cloudflare Workers run on V8 isolates with no
native Python/PyTorch support at all. Netlify Functions are Lambda-based — stateless,
short-lived, with package size limits well under what `torch` + `kokoro` + `openvoice` need
(several hundred MB before model weights). This backend needs a persistent server process
holding loaded models in memory — a "Web Service," not a serverless function.

Deploy the **frontend** (`App.jsx` / Vite build) to Netlify, Cloudflare Pages, or Vercel — all
fine for that half. Deploy **this backend** somewhere that runs a long-lived container:
Render, Railway, Fly.io, or a plain VPS.

### Docker

```bash
cd local-tts-server
docker build -t currenx-tts-backend .
docker run -p 8000:8000 --env-file ../.env currenx-tts-backend
```

Resource guidance:
- **RAM**: 2GB minimum, 4GB comfortable — torch + loaded Kokoro weights + OpenVoice converter +
  onnxruntime (via faster-whisper) adds up quickly.
- **Disk**: a few hundred MB (checkpoint + model caches), baked into the image at build time.
- **CPU-only is fine.** No GPU required — this runs on CPU already in local testing. Cloud CPU
  will simply be faster than a 2015 dual-core i5.

### Render / Railway

Point either platform at this Dockerfile directly (both support "Deploy from Dockerfile").
Set the same environment variables as the local `.env`. No `--reload` in production (already
handled — the Dockerfile's `CMD` omits it).

## Scaling considerations (if this grows beyond a single small instance)

- **Concurrency**: Kokoro + OpenVoice inference is CPU-bound and holds the event loop during
  generation. A single Uvicorn worker will serialize requests. For real concurrent load, run
  multiple worker processes (`uvicorn main:app --workers N`) or move generation into a task
  queue (Celery/RQ + Redis) so the API responds immediately and clients poll/webhook for
  completion.
- **Model loading cost**: both models load once at startup (a few seconds to tens of seconds).
  Keep the process warm — don't scale-to-zero aggressively, or every cold start re-pays that
  cost plus the checkpoint being read from disk.
- **GPU**: if generation speed becomes a bottleneck at scale, both Kokoro and OpenVoice support
  CUDA. Swapping `device="cpu"` for `device="cuda"` in the `ToneColorConverter` init, and
  ensuring `torch` is installed with CUDA support, would meaningfully speed up cloning
  specifically (the embedding extraction + conversion steps).
- **Caching**: identical (text, voice) pairs currently regenerate from scratch every time. A
  cache keyed on a hash of the request (checking `tts_logs` / `audio_generation_logs` in
  Supabase before generating) would cut redundant compute for repeated requests.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError` after any fresh install | Check `conda activate tts-server` / venv is actually active — a new terminal tab always resets to `base` |
| `_ARRAY_API not found` / torch import crash | NumPy 2.x + torch 2.2.2 ABI mismatch (Intel Mac only) — `conda install -c conda-forge "numpy<2"` |
| `[transformers] Disabling PyTorch because PyTorch >= 2.4 is required` | `transformers` version too new for your torch — pin `transformers==4.46.3` on Intel Mac, or use current versions on Linux where torch isn't capped |
| Clone request falls back to base preset silently | Check server logs for `[CLONE ENGINE WARNING]` — usually "input audio is too short" (see min-length section above) |
| Old cloned test audio appearing in the repo | OpenVoice's `se_extractor` cache (`processed/`) — gitignored now, safe to delete |
| `CondaHTTPError: CONNECTION FAILED` | Network hiccup during `conda install` — retry, or check for VPN/firewall interference with `conda.anaconda.org` |