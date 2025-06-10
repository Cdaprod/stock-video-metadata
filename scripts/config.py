# /scripts/config.py
import os
from pathlib import Path

# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm'}

# SFTP (BlackBox) credentials
SFTP_HOST   = os.getenv("SFTP_HOST", "sftp.blackboxglobal.com")
SFTP_PORT   = int(os.getenv("SFTP_PORT", 22))
SFTP_USER   = os.getenv("SFTP_USER", "username")
SFTP_PASS   = os.getenv("SFTP_PASS", "password")
REMOTE_ROOT = os.getenv("REMOTE_ROOT", "/incoming/videos")

def get_repo_root() -> Path:
    return Path.cwd()

def get_smb_root() -> Path:
    # adjust as needed
    return Path(r"B:/Video/StockFootage/Batches")