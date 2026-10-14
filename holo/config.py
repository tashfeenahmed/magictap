"""Paths and default tuning values."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
_MP = "https://storage.googleapis.com/mediapipe-models"
POSE_MODEL = (MODELS_DIR / "pose_landmarker_full.task",
              f"{_MP}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
HAND_MODEL = (MODELS_DIR / "hand_landmarker.task",
              f"{_MP}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
FACE_MODEL = (MODELS_DIR / "face_landmarker.task",
              f"{_MP}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
# ETH-XGaze ResNet-18 gaze model trained by hysts (ptgaze), on the Hugging Face Hub
GAZE_WEIGHTS = (MODELS_DIR / "eth_xgaze_resnet18.safetensors", ("hysts/ptgaze-eth-xgaze-resnet18", "model.safetensors"))

DATA_DIR = ROOT / "data"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
MODEL_PATH = DATA_DIR / "touch_model.joblib"
CELL_MODEL_PATH = DATA_DIR / "cell_model.joblib"
SESSIONS_DIR = DATA_DIR / "sessions"
RECORDINGS_DIR = DATA_DIR / "recordings"

# Camera
CAMERA_NAME = "MacBook"  # substring of the camera's name to pick by default
CAMERA_WIDTH = 1760  # 4:3 mode: taller view than 16:9, shows more of the arms
CAMERA_HEIGHT = 1328

# Body silhouette grid used as features (from the pose segmentation mask)
MASK_W, MASK_H = 32, 24

# Calibration timing (seconds)
REST_TIME = 8.0  # "don't touch" phase
HOLD_TIME = 1.0  # frames recorded per capture while the finger rests on the dot
