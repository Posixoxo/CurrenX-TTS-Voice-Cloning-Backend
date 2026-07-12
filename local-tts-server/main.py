import io
import os
import uuid
import tempfile
from datetime import datetime, timezone
from dotenv import load_dotenv
import torch
import requests

# Tell load_dotenv to look one folder up (the root folder) for the .env file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from supabase import create_client, Client

# OpenVoice V2 zero-shot tone color conversion
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

app = FastAPI(
    title="Kokoro Native Vector Matrix Server",
    description="Loads pre-selected celebrity .npy vector embeddings directly into the core neural layer, with OpenVoice V2 zero-shot cloning for user-uploaded samples.",
    version="5.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Supabase & ML Pipeline Initialization
# ----------------------------------------------------
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_SERVICE_ROLE_KEY")
SAMPLE_RATE = 24000  

if not SUPABASE_URL or not SUPABASE_KEY:
    print(f"\n[CRITICAL CONFIG ERROR] Could not read keys. Path location checked: {dotenv_path}")
    raise ValueError("Missing credentials in .env configuration.")

print("Connecting to Supabase Storage client...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase connection established successfully.")

print("Initializing Kokoro TTS Pipeline engine...")
pipeline = KPipeline(lang_code='a')
print("Kokoro TTS Pipeline loaded and ready.")

# OpenVoice V2 tone converter — loaded once at startup, guarded so a missing
# checkpoint never takes down the rest of the server (presets keep working).
OPENVOICE_CHECKPOINT_DIR = os.getenv("OPENVOICE_CHECKPOINT_DIR", "checkpoints_v2/converter")
tone_converter = None
try:
    print("Initializing OpenVoice V2 Tone Color Converter...")
    tone_converter = ToneColorConverter(
        f"{OPENVOICE_CHECKPOINT_DIR}/config.json",
        device="cpu"
    )
    tone_converter.load_ckpt(f"{OPENVOICE_CHECKPOINT_DIR}/checkpoint.pth")
    print("OpenVoice V2 converter loaded and ready — cloning enabled.")
except Exception as init_err:
    print(f"[WARNING] OpenVoice converter failed to initialize — cloning disabled, presets still work: {init_err}")

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    pitch_modifier: float = 1.0   
    speed_modifier: float = 1.0   

# ----------------------------------------------------
# Helper: Unified Voice Resolver, Blender & URL Interceptor
# ----------------------------------------------------
def resolve_voice_embedding(voice_input: str):
    """
    Intelligently handles custom celebrity profiles, runtime multi-vector combinations,
    and intercepts raw database audio storage URLs. For URLs this only picks a NEUTRAL
    BASE preset for Kokoro to generate with — actual identity cloning happens afterward
    via OpenVoice tone conversion, not here. This function's preset/blend logic for
    non-URL inputs is unchanged.
    """
    if not voice_input:
        return "af_heart"

    # --- 1. Handle Raw Database Storage URLs (clone requests) ---
    if voice_input.startswith("http://") or voice_input.startswith("https://"):
        print("[CLONE ENGINE] Detected uploaded sample URL. Generating neutral base pass for tone conversion.")
        if "male" in voice_input.lower() or "mayor" in voice_input.lower() or "adam" in voice_input.lower():
            return "am_adam"
        return "af_heart"

    # --- 2. Handle Explicit Plus Multi-Vector Combinations (e.g., "af_bella+af_nicole") ---
    if "+" in voice_input:
        print(f"[VECTOR ENGINE] Deconstructing runtime multi-vector mix: {voice_input}")
        try:
            voice_parts = [part.strip() for part in voice_input.split("+") if part.strip()]
            if not voice_parts:
                return "af_heart"
            
            loaded_tensors = []
            for part in voice_parts:
                try:
                    loaded_tensors.append(pipeline.load_voice(part))
                except Exception:
                    print(f"[VECTOR ENGINE WARNING] Failed loading sub-vector element: {part}")
            
            if not loaded_tensors:
                return "af_heart"
                
            blended_tensor = sum(loaded_tensors) / len(loaded_tensors)
            return blended_tensor
        except Exception as mix_err:
            print(f"[VECTOR ENGINE ERROR] Multi-vector compilation collapsed: {mix_err}")
            return "af_heart"

    # --- 3. Handle Custom Fixed Celebrity Blends ---
    if voice_input == "celebrity_jolie":
        print("[VECTOR ENGINE] Dynamically blending custom matrix for: Celebrity Jolie")
        try:
            v1 = pipeline.load_voice("af_heart")
            v2 = pipeline.load_voice("af_bella")
            return (v1 * 0.5) + (v2 * 0.5)
        except Exception:
            return "af_heart"

    elif voice_input == "celebrity_Mayor":
        print("[VECTOR ENGINE] Dynamically blending custom matrix for: Celebrity Mayor")
        try:
            v1 = pipeline.load_voice("am_adam")
            v2 = pipeline.load_voice("am_michael")
            return (v1 * 0.6) + (v2 * 0.4)
        except Exception:
            return "am_adam"
            
    # --- 4. Standard Native Premium Presets (e.g., "af_heart", "am_adam") ---
    print(f"[PRESET ENGINE] Routing standard premium voice preset: {voice_input}")
    return voice_input


def is_clone_request(voice_input: str) -> bool:
    """True when the frontend passed an uploaded sample URL instead of a preset/blend id."""
    return bool(voice_input) and (voice_input.startswith("http://") or voice_input.startswith("https://"))


# ----------------------------------------------------
# OpenVoice V2 Zero-Shot Cloning Support
# ----------------------------------------------------
def download_temp_audio(url: str) -> str:
    """Downloads the user's reference sample from Supabase Storage to a local temp wav file."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def clone_to_target_voice(base_wav_path: str, reference_wav_path: str, output_wav_path: str):
    """
    Zero-shot tone color conversion — no training loop. Extracts a speaker embedding
    from the reference clip and the base Kokoro output at inference time, then
    reshapes the base audio's timbre to match the reference. This is the piece
    that was previously a no-op fallback to a preset.
    """
    target_se, _ = se_extractor.get_se(reference_wav_path, tone_converter, vad=True)
    source_se, _ = se_extractor.get_se(base_wav_path, tone_converter, vad=True)
    tone_converter.convert(
        audio_src_path=base_wav_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_wav_path,
    )


def apply_openvoice_clone_if_requested(payload: "TTSRequest", binary_data: bytes) -> bytes:
    """
    Given the raw wav bytes Kokoro already produced, runs OpenVoice conversion
    against the uploaded reference sample if this was a clone request. Falls back
    silently to the original (preset) audio on any failure or if the converter
    isn't loaded, so a broken clone attempt never breaks generation entirely.
    """
    if not is_clone_request(payload.voice):
        return binary_data

    if tone_converter is None:
        print("[CLONE ENGINE WARNING] Clone requested but OpenVoice converter isn't loaded. Returning base preset audio.")
        return binary_data

    base_tmp_path = reference_tmp_path = cloned_tmp_path = None
    try:
        print("[CLONE ENGINE] Applying OpenVoice tone conversion to match uploaded sample...")

        base_tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(base_tmp_path, "wb") as f:
            f.write(binary_data)

        reference_tmp_path = download_temp_audio(payload.voice)
        cloned_tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        clone_to_target_voice(base_tmp_path, reference_tmp_path, cloned_tmp_path)

        with open(cloned_tmp_path, "rb") as f:
            cloned_bytes = f.read()

        print("[CLONE ENGINE] Tone conversion succeeded.")
        return cloned_bytes

    except Exception as clone_err:
        print(f"[CLONE ENGINE WARNING] Conversion failed, falling back to base preset audio: {clone_err}")
        return binary_data

    finally:
        for p in (base_tmp_path, reference_tmp_path, cloned_tmp_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass


# ----------------------------------------------------
# Native Pydub Client-Side Tuning Support Core
# ----------------------------------------------------
def apply_client_slider_adjustments(base_audio_np: np.ndarray, pitch_mod: float, speed_mod: float) -> np.ndarray:
    if pitch_mod == 1.0 and speed_mod == 1.0:
        return base_audio_np

    try:
        base_buffer = io.BytesIO()
        sf.write(base_buffer, base_audio_np, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        base_buffer.seek(0)
        segment = AudioSegment.from_wav(base_buffer)

        if pitch_mod != 1.0:
            target_sample_rate = int(segment.frame_rate * pitch_mod)
            segment = segment._spawn(segment.raw_data, overrides={'frame_rate': target_sample_rate})
            segment = segment.set_frame_rate(SAMPLE_RATE)

        if speed_mod != 1.0:
            segment = segment.speedup(playback_speed=speed_mod, chunk_size=150, crossfade=25)

        final_np = np.array(segment.get_array_of_samples(), dtype=np.float32)
        max_val = np.max(np.abs(final_np))
        if max_val > 0:
            final_np /= max_val

        return final_np
    except Exception as err:
        print(f"[DSP TUNING WARNING] Error processing array overlays: {err}")
        return base_audio_np

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------

@app.post("/api/tts/download")
async def generate_and_download_full_audio(payload: TTSRequest):
    if not pipeline or not supabase:
        raise HTTPException(status_code=500, detail="System configuration error.")
    
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="The 'text' field cannot be blank.")

    try:
        # Resolve voice model configuration structure
        resolved_voice = resolve_voice_embedding(payload.voice)

        print("\n[HYBRID PROCESS] Synthesizing crisp base neural speech...")
        
        # Branch invocation logic depending on voice return type
        if isinstance(resolved_voice, torch.Tensor):
            # Intelligent gender routing for Tensors using all known frontend payload indicators
            voice_lower = payload.voice.lower()
            if "am_" in voice_lower or "mayor" in voice_lower or "male" in voice_lower:
                pipeline_voice_preset = "am_adam"
            else:
                pipeline_voice_preset = "af_heart"
                
            generator = pipeline(
                payload.text, 
                voice=pipeline_voice_preset,
                speed=1.0, 
                split_pattern=r'\n+'
            )
            
            # Hot-swap matrix weights inside active session memory context
            if hasattr(pipeline, 'ctx') and hasattr(pipeline.ctx, 'style'):
                print(f"[VECTOR ENGINE] Hot-swapping context tensor over preset: {pipeline_voice_preset}")
                device = pipeline.ctx.style.device
                pipeline.ctx.style = resolved_voice.to(device).view(pipeline.ctx.style.shape)
        else:
            # Fall back to passing the string identifier directly for premium native presets
            generator = pipeline(
                payload.text, 
                voice=resolved_voice,
                speed=1.0, 
                split_pattern=r'\n+'
            )
        
        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Failed to synthesize base speech.")

        base_speech_audio = np.concatenate(audio_chunks)

        # OpenVoice's speaker-embedding extractor needs a minimum amount of audio
        # to work with (~4s observed threshold). Below that it either errors or
        # produces an unreliable embedding. Catch it here, before attempting
        # conversion, so the user gets a clear reason instead of a silent
        # fallback to the base preset voice.
        MIN_CLONE_AUDIO_SECONDS = 4.0
        base_duration_seconds = len(base_speech_audio) / SAMPLE_RATE
        if is_clone_request(payload.voice) and base_duration_seconds < MIN_CLONE_AUDIO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Text is too short for voice cloning ({base_duration_seconds:.1f}s of audio generated, "
                    f"need at least {MIN_CLONE_AUDIO_SECONDS:.0f}s). Please add more text and try again — "
                    f"roughly 80+ characters is usually enough."
                )
            )

        final_audio = apply_client_slider_adjustments(base_speech_audio, payload.pitch_modifier, payload.speed_modifier)

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, final_audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        binary_data = wav_buffer.read()

        # If the frontend passed an uploaded sample URL, reshape the base Kokoro
        # output to match that voice. No-op for regular preset/blend requests.
        binary_data = apply_openvoice_clone_if_requested(payload, binary_data)

        unique_id = str(uuid.uuid4())
        timestamp_str = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        storage_path = f"generations/{timestamp_str}/{unique_id}.wav"

        supabase.storage.from_("audio_generation_logs").upload(
            path=storage_path,
            file=binary_data,
            file_options={"content-type": "audio/wav", "x-upsert": "true"}
        )

        logged_voice_name = payload.voice if len(payload.voice) < 250 else payload.voice[:247] + "..."
        supabase.table("tts_logs").insert({
            "text_prompt": payload.text,
            "voice_used": logged_voice_name,
            "audio_duration_seconds": round(len(final_audio) / SAMPLE_RATE, 2),
            "storage_path": storage_path
        }).execute()

        output_buffer = io.BytesIO(binary_data)
        return StreamingResponse(
            output_buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={unique_id}.wav"}
        )

    except HTTPException:
        # Let intentional HTTPExceptions raised above (bad input, too-short
        # clone text, etc.) pass through with their real status code and
        # message instead of being flattened into a generic 500 below.
        raise
    except Exception as e:
        print(f"[SERVER EXCEPTION] Generation pipeline collapsed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation pipeline error: {str(e)}")


@app.post("/api/tts/stream")
async def stream_audio_chunks_realtime(payload: TTSRequest):
    """
    NOTE: Real-time cloning is not applied on this endpoint. OpenVoice's tone
    conversion needs the complete base clip to extract a speaker embedding, which
    is incompatible with chunk-by-chunk streaming. Clone requests (uploaded
    sample URLs) on this endpoint stream the neutral base preset only — route
    the frontend to /api/tts/download for actual cloned output.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="TTS Pipeline is uninitialized.")

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="The 'text' field cannot be blank.")

    if is_clone_request(payload.voice):
        print("[CLONE ENGINE] Clone requested on the streaming endpoint — not supported here. Streaming base preset instead.")

    # Resolve voice model configuration structure
    resolved_voice = resolve_voice_embedding(payload.voice)

    def audio_stream_generator():
        try:
            # Branch stream initialization logic depending on voice type context
            if isinstance(resolved_voice, torch.Tensor):
                voice_lower = payload.voice.lower()
                if "am_" in voice_lower or "mayor" in voice_lower or "male" in voice_lower:
                    pipeline_voice_preset = "am_adam"
                else:
                    pipeline_voice_preset = "af_heart"
                    
                generator = pipeline(
                    payload.text, 
                    voice=pipeline_voice_preset, 
                    speed=1.0, 
                    split_pattern=r'\n+'
                )
                
                # Hot-swap matrix weights inside active session memory context
                if hasattr(pipeline, 'ctx') and hasattr(pipeline.ctx, 'style'):
                    print(f"[VECTOR ENGINE STREAM] Hot-swapping context tensor over preset: {pipeline_voice_preset}")
                    device = pipeline.ctx.style.device
                    pipeline.ctx.style = resolved_voice.to(device).view(pipeline.ctx.style.shape)
            else:
                # Direct string route for native presets
                generator = pipeline(
                    payload.text, 
                    voice=resolved_voice, 
                    speed=1.0, 
                    split_pattern=r'\n+'
                )
            
            for _, _, audio in generator:
                if audio is not None and len(audio) > 0:
                    morphed_chunk = apply_client_slider_adjustments(audio, payload.pitch_modifier, payload.speed_modifier)
                    raw_buffer = io.BytesIO()
                    sf.write(raw_buffer, morphed_chunk, SAMPLE_RATE, format='RAW', subtype='PCM_16')
                    yield raw_buffer.getvalue()
                    
        except Exception as generator_error:
            print(f"Error encountered during stream yield: {generator_error}")
            yield b""

    return StreamingResponse(
        audio_stream_generator(), 
        media_type="audio/x-raw"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)