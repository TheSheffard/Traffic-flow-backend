import asyncio
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import httpx
import imageio_ffmpeg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import create_access_token, get_current_admin, public_admin, verify_password
from .database import DatabaseUnavailable, get_database, ping_database

load_dotenv()

DIRECT_UPLOAD_LIMIT = 4 * 1024 * 1024
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
VEHICLE_LABELS = {"car", "motorcycle", "motorbike", "bus", "truck"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


def configured_origins() -> list[str]:
    value = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


app = FastAPI(
    title="TrafficFlow AI Vercel API",
    description="Authenticated traffic analysis without local PyTorch/Ultralytics.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    email: str
    password: str


class VideoUrlRequest(BaseModel):
    video_url: str
    filename: str | None = None


def direction_for_box(box: dict) -> str:
    center_x = (float(box["xmin"]) + float(box["xmax"])) / 2
    center_y = (float(box["ymin"]) + float(box["ymax"])) / 2
    dx = center_x - FRAME_WIDTH / 2
    dy = center_y - FRAME_HEIGHT / 2
    if abs(dx) > abs(dy):
        return "East" if dx > 0 else "West"
    return "South" if dy > 0 else "North"


def extract_frames(video_path: Path, output_directory: Path) -> list[Path]:
    frame_limit = max(1, min(int(os.getenv("MAX_SAMPLED_FRAMES", "4")), 8))
    output_pattern = str(output_directory / "frame_%02d.jpg")
    filters = (
        "fps=1/2,"
        f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filters,
        "-frames:v",
        str(frame_limit),
        "-q:v",
        "4",
        output_pattern,
    ]
    try:
        subprocess.run(command, check=True, timeout=25, capture_output=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("The video took too long to decode. Use a shorter clip.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="ignore").strip()[:300]
        raise ValueError(f"The uploaded file could not be decoded as a video. {message}") from exc
    frames = sorted(output_directory.glob("frame_*.jpg"))
    if not frames:
        raise ValueError("No frames could be extracted from the video.")
    return frames


async def detect_frame(client: httpx.AsyncClient, frame_path: Path) -> list[dict]:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not configured in Vercel.")
    model = os.getenv("HF_MODEL", "facebook/detr-resnet-50").strip()
    base = os.getenv(
        "HF_ENDPOINT", "https://router.huggingface.co/hf-inference/models"
    ).strip().rstrip("/")
    response = await client.post(
        f"{base}/{model}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"},
        content=frame_path.read_bytes(),
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Detection provider returned {response.status_code}: {response.text[:300]}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("The detection provider returned an unexpected response.")
    return payload


async def analyze_frames(frames: list[Path]) -> dict[str, int]:
    counts = {"North": 0, "East": 0, "South": 0, "West": 0}
    threshold = float(os.getenv("DETECTION_THRESHOLD", "0.55"))
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
        results = await asyncio.gather(*(detect_frame(client, frame) for frame in frames))
    for detections in results:
        frame_counts = {direction: 0 for direction in counts}
        for detection in detections:
            label = str(detection.get("label", "")).lower()
            score = float(detection.get("score", 0))
            box = detection.get("box")
            if label not in VEHICLE_LABELS or score < threshold or not isinstance(box, dict):
                continue
            if not {"xmin", "ymin", "xmax", "ymax"}.issubset(box):
                continue
            frame_counts[direction_for_box(box)] += 1
        for direction, value in frame_counts.items():
            counts[direction] = max(counts[direction], value)
    return counts


def store_result(admin: dict, filename: str | None, counts: dict, sampled_frames: int) -> dict:
    result = {
        "admin_id": str(admin["_id"]),
        "admin_email": admin["email"],
        "filename": filename,
        "counts": counts,
        "total": sum(counts.values()),
        "sampled_frames": sampled_frames,
        "engine": "Hugging Face DETR",
        "created_at": datetime.now(timezone.utc),
    }
    ping_database()
    get_database().analysis_results.insert_one(result.copy())
    result["created_at"] = result["created_at"].isoformat()
    return result


async def analyze_path(path: Path, filename: str | None, admin: dict, directory: Path) -> dict:
    frames = await asyncio.to_thread(extract_frames, path, directory)
    counts = await analyze_frames(frames)
    return store_result(admin, filename, counts, len(frames))


def allowed_video_host(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    allowed = [item.strip().lower() for item in os.getenv("VIDEO_HOST_ALLOWLIST", "").split(",") if item.strip()]
    if parsed.scheme != "https" or not hostname:
        raise ValueError("video_url must use HTTPS.")
    if not allowed or not any(hostname == item or hostname.endswith(f".{item}") for item in allowed):
        raise ValueError("The video host is not included in VIDEO_HOST_ALLOWLIST.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("The video host could not be resolved.") from exc
    if any(ip_address(address).is_private or ip_address(address).is_loopback or ip_address(address).is_link_local for address in addresses):
        raise ValueError("Private network video URLs are not allowed.")
    return hostname


async def download_video(url: str, path: Path) -> None:
    allowed_video_host(url)
    max_bytes = max(1, int(os.getenv("MAX_REMOTE_VIDEO_MB", "50"))) * 1024 * 1024
    size = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise ValueError(f"The video host returned HTTP {response.status_code}.")
            with path.open("wb") as destination:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("The remote video is larger than the configured limit.")
                    destination.write(chunk)


@app.get("/")
def root() -> dict:
    return {"service": "TrafficFlow AI Vercel API", "status": "online", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    database_status = "connected"
    try:
        ping_database()
    except DatabaseUnavailable:
        database_status = "unavailable"
    return {
        "status": "ok",
        "database": database_status,
        "inference_configured": bool(os.getenv("HF_TOKEN", "").strip()),
    }


@app.post("/auth/login")
def login(payload: LoginRequest) -> dict:
    email = payload.email.strip().lower()
    try:
        ping_database()
        admin = get_database().admins.find_one({"email": email, "is_active": True})
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not admin or not verify_password(payload.password, admin.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    try:
        access_token, expires_in = create_access_token(str(admin["_id"]))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "admin": public_admin(admin),
    }


@app.get("/auth/me")
def me(admin: Annotated[dict, Depends(get_current_admin)]) -> dict:
    return public_admin(admin)


@app.post("/api/analyze-video")
async def analyze_video(
    admin: Annotated[dict, Depends(get_current_admin)],
    file: UploadFile = File(...),
) -> dict:
    suffix = Path(file.filename or "traffic.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload an MP4, MOV, WebM, AVI or MKV video.")
    with tempfile.TemporaryDirectory(prefix="trafficflow_") as temporary_directory:
        directory = Path(temporary_directory)
        video_path = directory / f"upload{suffix}"
        size = 0
        try:
            with video_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > DIRECT_UPLOAD_LIMIT:
                        raise HTTPException(
                            status_code=413,
                            detail="Direct uploads must be 4 MB or smaller on Vercel. For a larger video, upload it to storage and use /api/analyze-video-url.",
                        )
                    destination.write(chunk)
            return await analyze_path(video_path, file.filename, admin, directory)
        except HTTPException:
            raise
        except DatabaseUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
        finally:
            await file.close()


@app.post("/api/analyze-video-url")
async def analyze_video_url(
    payload: VideoUrlRequest,
    admin: Annotated[dict, Depends(get_current_admin)],
) -> dict:
    suffix = Path(urlparse(payload.video_url).path).suffix.lower() or ".mp4"
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="The URL must point to a supported video file.")
    with tempfile.TemporaryDirectory(prefix="trafficflow_") as temporary_directory:
        directory = Path(temporary_directory)
        video_path = directory / f"remote{suffix}"
        try:
            await download_video(payload.video_url, video_path)
            filename = payload.filename or Path(urlparse(payload.video_url).path).name
            return await analyze_path(video_path, filename, admin, directory)
        except DatabaseUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
