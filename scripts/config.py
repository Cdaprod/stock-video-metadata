# scripts/config.py
from dotenv import load_dotenv
load_dotenv()   # ← read .env into os.environ

import os
from pathlib import Path
from typing import List, Union

# --- Video file extensions ---
VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv',
    '.m4v', '.wmv', '.flv', '.webm'
}

# --- SFTP (BlackBox) credentials ---
SFTP_HOST   = os.getenv("SFTP_HOST", "sftp.blackboxglobal.com")
SFTP_PORT   = int(os.getenv("SFTP_PORT", 22))
SFTP_USER   = os.getenv("SFTP_USER", "username")
SFTP_PASS   = os.getenv("SFTP_PASS", "password")
REMOTE_ROOT = os.getenv("REMOTE_ROOT", "/incoming/videos")

# Local LLM & VTT model paths
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "")
LOCAL_VTT_PATH = os.getenv("LOCAL_VTT_PATH", "")

# --- Default SMB root paths ---
DEFAULT_SMB_ROOTS = [
    Path(r"B:/Video/StockFootage/Batches"),
    Path(r"B:/Video/StockFootage/Curated"),
]

# --- CONFIG object ---
CONFIG = {
    "VIDEO_EXTENSIONS": VIDEO_EXTENSIONS,
    "DEFAULT_OUTPUT_DIR": "./metadata",
    "DEFAULT_CURATION_DIR": "./curated",
    "BATCH_ROOT_PATHS": [],  # can be set dynamically at runtime
    "TRIM_POLICY": {
        # "clip1.mp4": (start_sec, duration_sec)
    },
}

# --- Functions ---

def get_repo_root() -> Path:
    return Path.cwd()

def get_smb_root() -> List[Path]:
    """Returns list of available SMB-style roots (or default fallback)."""
    return [p for p in DEFAULT_SMB_ROOTS if p.exists()]

def smb_available() -> bool:
    return any(p.exists() for p in get_smb_root())

def set_batch_paths(paths: Union[str, Path, List[Union[str, Path]]]):
    """Set one or more batch root paths (must actually exist)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    CONFIG["BATCH_ROOT_PATHS"] = [
        Path(p).expanduser().resolve()
        for p in paths
        if Path(p).expanduser().exists()
    ]

def get_batch_paths() -> List[Path]:
    """Returns configured or fallback batch root paths."""
    if CONFIG["BATCH_ROOT_PATHS"]:
        return CONFIG["BATCH_ROOT_PATHS"]
    return get_smb_root()
        
"""
# Example Notebook Cells:
from scripts import config

# manually set one or more paths
config.set_batch_paths(["B:/Video/Footage1", "B:/Video/Footage2"])

for p in config.get_batch_paths():
    print("📂 Found batch root:", p)
""" 

def interactive_batch_selection(base_path: Union[str, Path] = "."):
    """
    Walk the file tree from a base path and allow the user to select target directories.
    Requires: pip install inquirer
    """
    import inquirer
    base = Path(base_path).resolve()
    if not base.exists():
        print(f"❌ Base path does not exist: {base}")
        return

    choices = [
        str(p) for p in base.rglob("*")
        if p.is_dir() and not p.name.startswith(".")
    ]
    if not choices:
        print("⚠️ No subdirectories found.")
        return

    questions = [
        inquirer.Checkbox("selected_dirs",
                          message="Select directories to scan for video batches:",
                          choices=choices)
    ]
    answers = inquirer.prompt(questions)
    if answers and "selected_dirs" in answers:
        set_batch_paths(answers["selected_dirs"])
        print(f"✅ Set batch paths: {CONFIG['BATCH_ROOT_PATHS']}")

def should_include_directory(name: str, path: Path) -> bool:
    """Exclude hidden/system folders."""
    ignored = {"System Volume Information", "__MACOSX", ".DS_Store"}
    return not name.startswith('.') and name not in ignored

def should_include_video_file(path: Path) -> bool:
    """Basic filter for valid video files (>100KB)."""
    return (
        path.suffix.lower() in VIDEO_EXTENSIONS
        and path.stat().st_size > 100 * 1024
    )

def transform_batch_name(name: str) -> str:
    """Normalize raw directory names into batch labels."""
    return name.strip().replace(" ", "_").lower()

def get_output_path(filename: str) -> Path:
    """Generate output path inside metadata/."""
    return DEFAULT_OUTPUT_DIR / filename