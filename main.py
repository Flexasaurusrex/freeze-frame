"""
FREEZE FRAME by REWIND
Backend service for generating scroll-controlled music video experiences.

Pipeline:
1. User submits YouTube URL
2. yt-dlp downloads video at 720p
3. ffmpeg extracts audio → Opus
4. ffmpeg scene detection → key frames
5. Claude Vision API analyzes frames → selects 5 freeze moments
6. Frames optimized → WebP, served from /output
7. Frontend receives manifest JSON → renders scroll experience

Deploy on Railway with:
  - ANTHROPIC_API_KEY env var
  - PORT env var (Railway sets this automatically)
"""

import os
import json
import uuid
import asyncio
import subprocess
import shutil
import glob
import time
import math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))
BASE_URL = os.getenv("BASE_URL", "")  # e.g. https://freeze-frame-production.up.railway.app
OUTPUT_DIR = Path("output")
STATIC_DIR = Path("static")
MAX_VIDEO_DURATION = 600  # 10 minutes max
TARGET_FRAMES = 300
SAMPLE_FRAMES_FOR_AI = 20
FREEZE_POINTS = 5

# Ensure dirs exist
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Freeze Frame by REWIND")

# CORS — allow your REWIND domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wantmymtv.xyz",
        "https://www.wantmymtv.xyz",
        "https://cartoonrewind.com",
        "https://www.cartoonrewind.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "*",  # Remove in production if you want strict CORS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
# Serve generated outputs
app.mount("/output", StaticFiles(directory="output"), name="output")


# ═══════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════
class GenerateRequest(BaseModel):
    youtube_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "complete", "error"
    progress: int  # 0-100
    step: str  # Current step description
    manifest_url: Optional[str] = None
    error: Optional[str] = None


# In-memory job tracking (use Redis in production for persistence)
jobs: dict[str, dict] = {}


# ═══════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════

@app.get("/")
async def root():
    """Serve the Freeze Frame frontend"""
    return FileResponse("static/index.html")


@app.post("/api/generate")
async def generate_experience(req: GenerateRequest, bg: BackgroundTasks):
    """
    Start generating a Freeze Frame experience from a YouTube URL.
    Returns a job_id to poll for progress.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured")

    # Validate YouTube URL
    video_id = extract_video_id(req.youtube_url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL")

    # Check if we already generated this video
    existing = OUTPUT_DIR / video_id / "manifest.json"
    if existing.exists():
        return {
            "job_id": video_id,
            "status": "complete",
            "manifest_url": f"/output/{video_id}/manifest.json"
        }

    # Create job
    job_id = video_id
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "step": "Starting...",
        "manifest_url": None,
        "error": None,
    }

    # Run pipeline in background
    bg.add_task(run_pipeline, job_id, req.youtube_url, video_id)

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll for job progress"""
    if job_id not in jobs:
        # Check if manifest exists (completed previously)
        existing = OUTPUT_DIR / job_id / "manifest.json"
        if existing.exists():
            return {
                "job_id": job_id,
                "status": "complete",
                "progress": 100,
                "step": "Complete",
                "manifest_url": f"/output/{job_id}/manifest.json"
            }
        raise HTTPException(404, "Job not found")

    return {"job_id": job_id, **jobs[job_id]}


@app.get("/api/experience/{video_id}")
async def get_experience(video_id: str):
    """Get a completed experience manifest"""
    manifest_path = OUTPUT_DIR / video_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Experience not found")

    with open(manifest_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════

def update_job(job_id: str, progress: int, step: str,
               status: str = "processing", error: str = None,
               manifest_url: str = None):
    if job_id in jobs:
        jobs[job_id].update({
            "progress": progress,
            "step": step,
            "status": status,
            "error": error,
            "manifest_url": manifest_url,
        })


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats"""
    import re
    patterns = [
        r'(?:v=|youtu\.be\/|shorts\/)([\w-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


async def run_pipeline(job_id: str, youtube_url: str, video_id: str):
    """Main processing pipeline"""
    work_dir = OUTPUT_DIR / video_id
    work_dir.mkdir(exist_ok=True)
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    try:
        # ── Step 1: Get video info ──
        update_job(job_id, 5, "Fetching video metadata...")
        info = await get_video_info(youtube_url)
        if not info:
            raise Exception("Could not fetch video info")

        title = info.get("title", "Unknown")
        channel = info.get("channel", "Unknown")
        duration = info.get("duration", 0)

        if duration > MAX_VIDEO_DURATION:
            raise Exception(f"Video too long ({duration}s). Max is {MAX_VIDEO_DURATION}s.")

        update_job(job_id, 10, f"Found: {title[:50]}...")

        # ── Step 2: Download video ──
        update_job(job_id, 15, "Downloading video (720p)...")
        video_path = work_dir / "video.mp4"
        await download_video(youtube_url, str(video_path))
        update_job(job_id, 35, "Download complete")

        # ── Step 3: Extract audio ──
        update_job(job_id, 38, "Extracting audio track...")
        audio_path = work_dir / "audio.opus"
        # Also create mp3 for broader browser support
        audio_mp3_path = work_dir / "audio.mp3"
        await extract_audio(str(video_path), str(audio_path), str(audio_mp3_path))
        update_job(job_id, 45, "Audio extracted")

        # ── Step 4: Extract key frames ──
        update_job(job_id, 48, "Running scene detection...")
        await extract_frames(str(video_path), str(frames_dir), TARGET_FRAMES)
        update_job(job_id, 60, f"Extracted {TARGET_FRAMES} key frames")

        # ── Step 5: AI Analysis ──
        update_job(job_id, 65, "Sending frames to Claude Vision API...")
        freeze_moments = await analyze_frames_with_claude(
            str(frames_dir), title, channel, TARGET_FRAMES
        )
        update_job(job_id, 82, f"AI identified {len(freeze_moments)} freeze moments")

        # ── Step 6: Optimize frames to WebP ──
        update_job(job_id, 85, "Optimizing frames → WebP...")
        await optimize_frames(str(frames_dir))
        update_job(job_id, 92, "Frames optimized")

        # ── Step 7: Generate manifest ──
        update_job(job_id, 95, "Generating experience manifest...")

        # Get list of frame files
        frame_files = sorted(glob.glob(str(frames_dir / "*.webp")))
        if not frame_files:
            # Fallback to jpg if webp conversion failed
            frame_files = sorted(glob.glob(str(frames_dir / "*.jpg")))

        frame_urls = [
            f"/output/{video_id}/frames/{Path(f).name}"
            for f in frame_files
        ]

        # Determine audio URL (prefer mp3 for compatibility)
        audio_url = None
        if audio_mp3_path.exists():
            audio_url = f"/output/{video_id}/audio.mp3"
        elif audio_path.exists():
            audio_url = f"/output/{video_id}/audio.opus"

        manifest = {
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "duration": duration,
            "total_frames": len(frame_urls),
            "frame_urls": frame_urls,
            "audio_url": audio_url,
            "freeze_moments": freeze_moments,
            "generated_at": int(time.time()),
            "intro_frames": 90,  # First 90 frames = REWIND branded intro
        }

        manifest_path = work_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Clean up video file (keep frames + audio)
        if video_path.exists():
            video_path.unlink()

        update_job(
            job_id, 100, "Experience ready!",
            status="complete",
            manifest_url=f"/output/{video_id}/manifest.json"
        )

    except Exception as e:
        update_job(job_id, 0, str(e), status="error", error=str(e))
        # Clean up on error
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ═══════════════════════════════════════════
# PIPELINE HELPERS
# ═══════════════════════════════════════════

async def get_video_info(url: str) -> Optional[dict]:
    """Get video metadata using yt-dlp"""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                url,
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                "title": data.get("title", "Unknown"),
                "channel": data.get("channel", data.get("uploader", "Unknown")),
                "duration": data.get("duration", 0),
                "thumbnail": data.get("thumbnail", ""),
            }
    except Exception as e:
        print(f"Error getting video info: {e}")
    return None


async def download_video(url: str, output_path: str):
    """Download video at 720p using yt-dlp"""
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--no-playlist",
            url,
        ],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise Exception(f"yt-dlp failed: {result.stderr[:200]}")


async def extract_audio(video_path: str, opus_path: str, mp3_path: str):
    """Extract audio track from video"""
    # Extract as MP3 for max browser compatibility
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-b:a", "128k",
            "-ar", "44100",
            mp3_path,
        ],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"MP3 extraction warning: {result.stderr[:200]}")

    # Also try Opus for smaller size
    try:
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "libopus",
                "-b:a", "128k",
                opus_path,
            ],
            capture_output=True, text=True, timeout=120
        )
    except Exception:
        pass  # Opus is optional, MP3 is primary


async def extract_frames(video_path: str, frames_dir: str, target_count: int):
    """
    Extract key frames using ffmpeg scene detection.
    Falls back to uniform sampling if scene detection doesn't yield enough frames.
    """
    # First: try scene-based extraction
    result = await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"select=gt(scene\\,0.03),scale=1280:720",
            "-vsync", "vfr",
            "-q:v", "2",
            f"{frames_dir}/scene_%04d.jpg",
        ],
        capture_output=True, text=True, timeout=180
    )

    scene_frames = sorted(glob.glob(f"{frames_dir}/scene_*.jpg"))

    if len(scene_frames) >= target_count:
        # Too many scene frames — subsample evenly
        step = len(scene_frames) / target_count
        selected = [scene_frames[int(i * step)] for i in range(target_count)]
        # Rename to sequential
        for i, src in enumerate(selected):
            dst = f"{frames_dir}/frame_{i:04d}.jpg"
            if src != dst:
                shutil.move(src, dst)
        # Clean up extras
        for f in glob.glob(f"{frames_dir}/scene_*.jpg"):
            os.remove(f)
    elif len(scene_frames) < target_count // 2:
        # Not enough scene frames — fall back to uniform extraction
        # Clean scene frames
        for f in scene_frames:
            os.remove(f)

        # Get duration
        probe = await asyncio.to_thread(
            subprocess.run,
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, timeout=30
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        fps = target_count / duration

        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"fps={fps:.4f},scale=1280:720",
                "-q:v", "2",
                f"{frames_dir}/frame_%04d.jpg",
            ],
            capture_output=True, text=True, timeout=180
        )
    else:
        # Roughly right number — just rename
        for i, f in enumerate(scene_frames[:target_count]):
            shutil.move(f, f"{frames_dir}/frame_{i:04d}.jpg")
        for f in glob.glob(f"{frames_dir}/scene_*.jpg"):
            os.remove(f)


async def optimize_frames(frames_dir: str):
    """Convert JPG frames to WebP for smaller size"""
    jpg_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
    for jpg in jpg_files:
        webp = jpg.replace(".jpg", ".webp")
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y",
                "-i", jpg,
                "-quality", "80",
                webp,
            ],
            capture_output=True, text=True, timeout=10
        )
        if os.path.exists(webp):
            os.remove(jpg)  # Remove original JPG


async def analyze_frames_with_claude(
    frames_dir: str, title: str, channel: str, total_frames: int
) -> list[dict]:
    """
    Send sample frames to Claude Vision API to identify
    the 5 most visually iconic moments for freeze frames.
    """
    import base64

    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg")) + \
                  sorted(glob.glob(f"{frames_dir}/frame_*.webp"))

    if not frame_files:
        # Return default freeze moments if no frames
        return default_freeze_moments()

    # Sample evenly across the video (skip first 10% for intro variety)
    start_idx = int(len(frame_files) * 0.1)
    sample_indices = [
        int(start_idx + (i / SAMPLE_FRAMES_FOR_AI) * (len(frame_files) - start_idx))
        for i in range(SAMPLE_FRAMES_FOR_AI)
    ]
    sample_indices = [min(i, len(frame_files) - 1) for i in sample_indices]

    # Create a mapping of sample index to actual frame index for reference
    frame_index_map = {i: sample_indices[i] for i in range(len(sample_indices))}

    # Build message content with images
    content = [
        {
            "type": "text",
            "text": f"""You are a visual curator for REWIND — a platform that transforms music videos into scroll-controlled cinematic experiences. You have an extraordinary eye for the frames that define a music video's visual identity.

You're analyzing frames from "{title}" by {channel}.

I'm showing you {SAMPLE_FRAMES_FOR_AI} key frames from this video. Each frame is labeled as "SAMPLE #X" - this is the reference number you'll use.

YOUR TASK: Select the 5 most visually POWERFUL frames from the ones I'm showing you. Think like a museum curator choosing which frames deserve to be frozen in time.

CRITICAL: You can ONLY select from the {SAMPLE_FRAMES_FOR_AI} frames I'm showing you. Reference them by their SAMPLE number (0-{SAMPLE_FRAMES_FOR_AI - 1}).

Look for:
- Dramatic lighting shifts or color explosions
- Peak emotional expressions or body language
- Iconic composition (symmetry, leading lines, silhouettes)
- Visual turning points where the aesthetic transforms
- The single frame that could be the album cover

For each of the 5 moments, provide:
- sample_index: Which sample frame (0-{SAMPLE_FRAMES_FOR_AI - 1}) - REQUIRED
- title: A powerful, evocative title (2-5 words, ALL CAPS) — think opening credits or chapter titles
- description: 2-3 sentences of CINEMATIC COMMENTARY on THIS SPECIFIC FRAME. Write like a film director analyzing a critical shot — describe the visual composition, the emotional weight, the narrative moment. What makes THIS frame unforgettable? What story does it tell in a single instant? Make it VIVID and IMPACTFUL.

IMPORTANT:
- Space the 5 selections across the video (don't cluster them)
- Your description must match the EXACT frame you're selecting
- Write descriptions that HIT HARD — this is visual storytelling at its peak
- Only select from the frames I showed you

Respond ONLY with a JSON array:
[
  {{"sample_index": 3, "title": "THE OPENING SHOT", "description": "..."}},
  ...
]"""
        }
    ]

    # Add sample frames as images
    for sample_idx, frame_idx in enumerate(sample_indices):
        frame_path = frame_files[frame_idx]
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        ext = Path(frame_path).suffix.lower()
        media_type = "image/webp" if ext == ".webp" else "image/jpeg"

        content.append({
            "type": "text",
            "text": f"SAMPLE #{sample_idx}:"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            }
        })

    # Call Claude Vision API
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-5-20250929",  # Best vision quality for iconic moment selection
                    "max_tokens": 2048,  # More tokens for longer, cinematic descriptions
                    "messages": [
                        {"role": "user", "content": content}
                    ],
                },
            )

            if response.status_code != 200:
                print(f"Claude API error: {response.status_code} {response.text[:300]}")
                return default_freeze_moments()

            data = response.json()
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]

            # Parse JSON from response
            # Strip any markdown code fences
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            moments = json.loads(text)

            # Validate and normalize - map sample_index back to actual position
            result = []
            for m in moments[:FREEZE_POINTS]:
                sample_idx = int(m.get("sample_index", 0))

                # Ensure sample_idx is valid
                if sample_idx < 0 or sample_idx >= len(sample_indices):
                    print(f"Invalid sample_index {sample_idx}, skipping")
                    continue

                # Get the actual frame index from our sample
                actual_frame_idx = sample_indices[sample_idx]

                # Calculate position in video (0.0 to 1.0)
                position = actual_frame_idx / total_frames

                result.append({
                    "at": max(0.05, min(0.95, position)),
                    "title": str(m.get("title", "FREEZE FRAME"))[:50],
                    "desc": str(m.get("description", "A frozen moment in time."))[:500],
                })

            if len(result) >= 3:
                return result

    except Exception as e:
        print(f"Claude Vision analysis error: {e}")

    return default_freeze_moments()


def default_freeze_moments() -> list[dict]:
    """Fallback freeze moments if AI analysis fails"""
    return [
        {"at": 0.12, "title": "THE OPENING SHOT", "desc": "Where every great story begins — a single frame that sets the mood for everything to come."},
        {"at": 0.32, "title": "THE CRESCENDO", "desc": "The moment the beat drops and the visual language shifts. Colors explode."},
        {"at": 0.52, "title": "THE TURNING POINT", "desc": "The frame where the narrative pivots and nothing is the same."},
        {"at": 0.72, "title": "THE ICONIC FRAME", "desc": "The one everyone shares. The frame that defines the era."},
        {"at": 0.92, "title": "THE FINAL CHORD", "desc": "The last breath before fade to black."},
    ]


# ═══════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
