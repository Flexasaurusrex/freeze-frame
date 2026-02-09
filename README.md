# FREEZE FRAME by REWIND

Scroll-controlled music video experience generator. Drop a YouTube URL, AI analyzes the video, and generates an interactive scroll-controlled cinematic experience with freeze frame moments.

## Architecture

```
User pastes YouTube URL
        ↓
  Frontend → POST /api/generate
        ↓
  Backend Pipeline:
  1. yt-dlp downloads video (720p)
  2. ffmpeg extracts audio → MP3
  3. ffmpeg scene detection → 300 key frames
  4. Claude Vision API (Haiku 4.5) → 5 freeze moments
  5. Frames optimized → WebP
  6. Manifest JSON generated
        ↓
  Frontend polls GET /api/status/{job_id}
        ↓
  Frontend loads manifest → renders scroll experience
  Audio plays continuously in background
  Visuals controlled by scroll position
```

## Deploy on Railway

### 1. Create new service
- Go to Railway dashboard
- Click "+ New" → "GitHub Repo"
- Select your repo
- Railway will detect the Dockerfile

### 2. Set environment variables
In Railway → Variables tab, add:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

That's it. Railway auto-sets `PORT`.

### 3. Generate domain
- Settings → Networking → Generate Domain
- You'll get: `freeze-frame-production.up.railway.app`

### 4. Update frontend
Set `API_BASE` in `static/index.html` to your Railway URL.

## Local Development

```bash
# Install deps
pip install -r requirements.txt

# Install system tools
brew install ffmpeg yt-dlp  # macOS
# or: apt install ffmpeg && pip install yt-dlp

# Set API key
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Run
python main.py
```

Server runs at http://localhost:8080

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve frontend |
| POST | `/api/generate` | Start generation (body: `{youtube_url}`) |
| GET | `/api/status/{job_id}` | Poll job progress |
| GET | `/api/experience/{video_id}` | Get completed manifest |
| GET | `/output/{video_id}/...` | Serve frames/audio |

## Cost Per Generation

- Claude Sonnet 4.5 Vision: ~$0.05-0.08 (20 images × ~1,200 tokens each)
- Storage: ~3-5MB per experience (300 WebP frames + audio)
- Processing time: ~30-60 seconds

## Tech Stack

- **Backend:** Python FastAPI
- **Video:** yt-dlp + ffmpeg
- **AI:** Claude Sonnet 4.5 Vision API
- **Frontend:** Vanilla HTML/CSS/JS with Canvas API
- **Deploy:** Railway (Docker)
