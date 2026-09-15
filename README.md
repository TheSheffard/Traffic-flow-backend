# TrafficFlow AI — Lightweight Vercel Backend v2

This package keeps the original backend features—administrator login, JWT sessions,
MongoDB analysis history, video decoding, vehicle detection, and North/East/South/West
traffic counts—without bundling Ultralytics, PyTorch, OpenCV, or a local YOLO weight.

The video is decoded with a small FFmpeg binary. Four sampled JPEG frames are sent to
Hugging Face's hosted DETR object-detection model. The API response remains compatible
with the original frontend.

## Why the original failed

The `yolo11n.pt` file is only about 5.4 MB. The real deployment size came from
`ultralytics`, which installs PyTorch and other large native dependencies. Those packages
can push a Vercel Python function over its bundle limit.

## Deploy on Vercel

1. Extract this ZIP into a new GitHub repository. `app/main.py` is the FastAPI entrypoint.
2. Import that repository into Vercel. Leave Root Directory blank if these files are at
   the repository root.
3. In **Vercel → Project → Settings → Environment Variables**, add:

```text
FRONTEND_ORIGIN=https://your-frontend.vercel.app
MONGODB_URI=your MongoDB Atlas URI
MONGODB_DATABASE=trafficflow_ai
JWT_SECRET=a unique random value containing at least 32 characters
JWT_EXPIRES_MINUTES=480
HF_TOKEN=your Hugging Face token
HF_MODEL=facebook/detr-resnet-50
DETECTION_THRESHOLD=0.55
MAX_SAMPLED_FRAMES=4
```

Get an inference-enabled token from `https://huggingface.co/settings/tokens`. Never put
tokens, passwords, or a real `.env` file in GitHub. Redeploy after saving variables.

4. Set the frontend variable to your new backend URL and redeploy the frontend:

```text
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

5. Check `https://your-backend.vercel.app/health`. It should report `database` as
   `connected` and `inference_configured` as `true`.

## Administrator account

The admin collection and password format are unchanged. An existing MongoDB admin should
continue to work. To create or update one from your computer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.seed_admin
```

Put your real MongoDB URI and admin details in the local `.env` before running the last
command. The `.gitignore` prevents that file from being committed.

## API compatibility

- `POST /auth/login` — unchanged.
- `GET /auth/me` — unchanged; send `Authorization: Bearer TOKEN`.
- `POST /api/analyze-video` — unchanged form field (`file`) and result structure. Because
  Vercel Functions cap request bodies at 4.5 MB, the file must be 4 MB or smaller.
- `POST /api/analyze-video-url` — added for larger files. Upload the video to Vercel Blob,
  Cloudinary, S3, or another HTTPS host, then send its URL. Add the storage hostname to
  `VIDEO_HOST_ALLOWLIST` first.
- `GET /health` — checks MongoDB and hosted-inference configuration.

Example result:

```json
{
  "admin_id": "...",
  "admin_email": "admin@example.com",
  "filename": "traffic.mp4",
  "counts": {"North": 3, "East": 7, "South": 4, "West": 5},
  "total": 19,
  "sampled_frames": 4,
  "engine": "Hugging Face DETR",
  "created_at": "2026-09-15T20:00:00+00:00"
}
```

For a large-video request:

```json
POST /api/analyze-video-url
Authorization: Bearer TOKEN
Content-Type: application/json

{"video_url":"https://your-allowed-host.example/traffic.mp4"}
```

## Important limits

- Direct request/response payloads on Vercel Functions are limited to 4.5 MB.
- Remote videos default to 50 MB and can be adjusted with `MAX_REMOTE_VIDEO_MB`.
- The system samples up to four frames and returns peak visible density per direction; it
  does not track unique vehicles across the entire video.
- Hosted inference requires network access and may be rate-limited by the provider.

