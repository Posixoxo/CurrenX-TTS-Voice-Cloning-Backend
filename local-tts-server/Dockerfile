FROM python:3.11-slim

# System dependencies:
# - libsndfile1: required by `soundfile` for reading/writing WAV
# - ffmpeg: required by `pydub` for audio segment manipulation
# - build-essential: some pinned wheels (e.g. wavmark, jieba) fall back to
#   source builds on certain platforms; keep this until confirmed unnecessary
# - git: required to pip-install OpenVoice directly from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first, separately from app code, so Docker's layer
# cache skips this ~5-10 minute step on every code-only change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Bake the OpenVoice V2 checkpoint into the image at build time. This avoids
# a slow/fragile download on every container start, and works around
# ephemeral filesystems on platforms like Render/Railway that don't persist
# disk writes between deploys.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='checkpoints_v2')"

# App code last, since this changes most often.
COPY . .

ENV OPENVOICE_CHECKPOINT_DIR=checkpoints_v2/converter
ENV PORT=8000

EXPOSE 8000

# No --reload in production — that's a dev-only convenience flag.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]