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
    """Returns list of available SMB-style roots (or default fallback)"""
    paths = os.getenv("SMB_ROOTS")
    if paths:
        return [Path(p.strip()).expanduser() for p in paths.split(",")]
    return DEFAULT_SMB_ROOTS

def smb_available() -> bool:
    return any(p.exists() for p in get_smb_root())

def set_batch_paths(paths: Union[str, Path, List[Union[str, Path]]]):
    """Set one or more batch root paths"""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    CONFIG["BATCH_ROOT_PATHS"] = [Path(p).expanduser().resolve() for p in paths if Path(p).exists()]

def get_batch_paths() -> List[Path]:
    """Returns configured or fallback batch root paths"""
    if CONFIG["BATCH_ROOT_PATHS"]:
        return CONFIG["BATCH_ROOT_PATHS"]
    return [p for p in get_smb_root() if p.exists()]
    
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
    """
    import inquirer  # pip install inquirer
    base_path = Path(base_path).resolve()
    if not base_path.exists():
        print(f"❌ Base path does not exist: {base_path}")
        return

    # Gather all subdirs excluding ignored
    choices = [
        str(p) for p in base_path.rglob("*")
        if p.is_dir() and not any(part in CONFIG["IGNORE_DIRECTORIES"] for part in p.parts)
    ]
    if not choices:
        print("⚠️ No subdirectories found.")
        return

    questions = [
        inquirer.Checkbox(
            "selected_dirs",
            message="Select directories to scan for video batches:",
            choices=choices,
        )
    ]
    answers = inquirer.prompt(questions)
    if answers and "selected_dirs" in answers:
        set_batch_paths(answers["selected_dirs"])
        print(f"✅ Set batch paths: {CONFIG['BATCH_ROOT_PATHS']}")

"""
# Example Notebook Cell:
from config import interactive_batch_selection, save_workspace_info

interactive_batch_selection("B:/Video/StockFootage")
save_workspace_info()
"""

# Add to scripts/config.py

def should_include_directory(name: str, path: Path) -> bool:
    """Exclude hidden/system folders, allow customization."""
    ignored = {"System Volume Information", ".DS_Store", "__MACOSX"}
    return not name.startswith('.') and name not in ignored

def should_include_video_file(path: Path) -> bool:
    """Basic filter for valid video files."""
    return path.suffix.lower() in VIDEO_EXTENSIONS and path.stat().st_size > 1024 * 100  # >100KB

def transform_batch_name(name: str) -> str:
    """Transform raw directory names into normalized batch labels."""
    return name.strip().replace(" ", "_").lower()

def get_output_path(filename: str) -> Path:
    """Generate output path inside metadata/."""
    return Path(CONFIG["DEFAULT_OUTPUT_DIR"]) / filename