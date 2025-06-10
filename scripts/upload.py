# /scripts/upload.py
import os
import zipfile
from pathlib import Path

# Attempt to import Paramiko (SFTP). If unavailable, we'll fall back to ZIP packaging.
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    print("⚠️  Paramiko not installed. SFTP upload disabled; will create ZIP archives instead.")

# Absolute import from scripts/config
from config import SFTP_HOST, SFTP_PORT, SFTP_USER, SFTP_PASS, REMOTE_ROOT

def sftp_connect():
    if not HAS_PARAMIKO:
        raise RuntimeError("Paramiko is not available. Cannot establish SFTP connection.")
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USER, password=SFTP_PASS)
    return paramiko.SFTPClient.from_transport(transport)

def upload_batch_or_zip(batch_path: Path, zip_output_dir: Path):
    """
    Attempts to upload the batch directory via SFTP. If SFTP is not available
    or if upload fails, falls back to creating a ZIP archive in zip_output_dir.
    """
    batch_name = batch_path.name

    # Prepare fallback ZIP directory
    zip_output_dir.mkdir(parents=True, exist_ok=True)

    # Try SFTP upload
    if HAS_PARAMIKO and all([SFTP_HOST, SFTP_USER, SFTP_PASS]):
        try:
            sftp = sftp_connect()
            remote_folder = f"{REMOTE_ROOT}/{batch_name}"
            try:
                sftp.mkdir(remote_folder)
            except IOError:
                pass  # folder already exists

            # Upload metadata.xml + all video files
            for file in batch_path.iterdir():
                if file.is_file():
                    remote_path = f"{remote_folder}/{file.name}"
                    sftp.put(str(file), remote_path)
                    print(f"📤 Uploaded {file.name} to {remote_folder}")

            sftp.close()
            print(f"✅ SFTP upload succeeded for batch `{batch_name}`")
            return
        except Exception as e:
            print(f"❌ SFTP upload failed for `{batch_name}`: {e}")
            print("📦 Falling back to ZIP packaging...")

    # Fallback to ZIP
    zip_path = zip_output_dir / f"{batch_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in batch_path.iterdir():
            if file.is_file():
                zf.write(file, arcname=f"{batch_name}/{file.name}")
    print(f"📦 Created ZIP for manual upload: {zip_path}")