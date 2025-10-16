import os
from dotenv import load_dotenv
from typing import Dict

env_path = os.path.join(os.path.expanduser("~"), "ai-drone-ws", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    raise FileNotFoundError(f".env file not found at {env_path}")

def strip_comment(s: str) -> str:
    return s.split("#", 1)[0].strip()

def require_env(key: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return strip_comment(raw)

def as_bool(s: str) -> bool:
    return strip_comment(s).lower() in ("1", "true", "yes", "on")

def parse_class_map(s: str) -> Dict[int, str]:
    s = strip_comment(s)
    out: Dict[int, str] = {}
    if not s:
        return out
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split(":", 1)
        out[int(k.strip())] = v.strip()
    return out

# --- YOLO / TRT ---
PT_PATH       = require_env("PT_PATH")
CONF_TH       = float(require_env("CONF_TH"))
KEEP_CLASSES  = tuple(int(x) for x in require_env("KEEP_CLASSES").replace(" ", "").split(","))
IMG_SZ_GPU    = int(require_env("IMG_SZ_GPU"))
IMG_SZ_CPU    = int(require_env("IMG_SZ_CPU"))
NMS_IOU       = float(require_env("NMS_IOU"))

ONNX_PATH     = os.getenv("ONNX_PATH")
ENGINE_PATH   = os.getenv("ENGINE_PATH")
YOLO_INPUT_W  = os.getenv("YOLO_INPUT_W")
YOLO_INPUT_H  = os.getenv("YOLO_INPUT_H")
YOLO_CONF_TH  = os.getenv("YOLO_CONF_TH")
YOLO_CLASSES  = os.getenv("YOLO_CLASSES")

# --- RTSP ---
RTSP_URL       = require_env("RTSP_URL")
VIDEO_WIDTH    = int(require_env("VIDEO_WIDTH"))
VIDEO_HEIGHT   = int(require_env("VIDEO_HEIGHT"))
VIDEO_FPS      = int(require_env("VIDEO_FPS"))
VIDEO_LATENCY  = int(os.getenv("VIDEO_LATENCY", "0"))
VIDEO_USE_HW   = os.getenv("VIDEO_USE_HW", "false").lower() in ("true", "1", "yes")
VIDEO_ROTATE   = int(require_env("VIDEO_ROTATE"))

# --- Video file ---
VIDEO_FILE     = require_env("VIDEO_FILE")
LOOP_VIDEO     = as_bool(require_env("LOOP_VIDEO"))
RESIZE_W       = int(require_env("RESIZE_W"))
RESIZE_H       = int(require_env("RESIZE_H"))
ROTATE_DEG     = int(require_env("ROTATE_DEG"))
MIRROR         = as_bool(require_env("MIRROR"))
SPEED_FACTOR   = float(require_env("SPEED_FACTOR"))
JPEG_QUALITY   = int(require_env("JPEG_QUALITY"))
PUB_RAW_TOPIC  = require_env("PUB_RAW_TOPIC")
PUB_COMP_TOPIC = require_env("PUB_COMP_TOPIC")

# --- Tracker ---
MAX_AGE  = int(require_env("MAX_AGE"))
DIST_TH  = float(require_env("DIST_TH"))
DIST_TH2 = DIST_TH ** 2

# --- Telemetry ---
_ORIGIN_LAT = float(os.getenv("TELEM_ORIGIN_LAT", "48.8566"))
_ORIGIN_LON = float(os.getenv("TELEM_ORIGIN_LON", "2.3522"))
_ORIGIN_ALT = float(os.getenv("TELEM_ORIGIN_ALT", "35.0"))
_YAW_OFFSET = float(os.getenv("TELEM_YAW_OFFSET", "0.0"))

# --- Flask/stream ---
STREAM_TOPIC         = os.getenv("STREAM_TOPIC", "/camera/overlay/compressed")
FLASK_HOST           = "0.0.0.0"
FLASK_PORT           = int(os.getenv("FLASK_PORT", "5000"))
FLASK_QUALITY        = int(os.getenv("FLASK_QUALITY", "80"))
READY_TIMEOUT_S      = float(os.getenv("READY_TIMEOUT", "3.0"))
DEFAULT_STREAM_TOPIC = "/camera/overlay/compressed"
DEFAULT_PORT         = 5000

# --- Camera geometry ---
IMG_W        = int(require_env("IMG_W"))
IMG_H        = int(require_env("IMG_H"))
HFOV_DEG     = float(require_env("HFOV_DEG"))
VFOV_DEG     = float(require_env("VFOV_DEG"))
MIN_TAN      = float(require_env("MIN_TAN"))
MAX_RANGE_M  = float(require_env("MAX_RANGE_M"))
STATUS_HZ    = float(require_env("STATUS_HZ"))

# --- Class map ---
CLASS_MAP: Dict[int, str] = parse_class_map(require_env("CLASS_MAP"))
