import os
import uuid
from pathlib import Path
from fastapi import UploadFile
from app.config import settings
 
 
def ensure_upload_dir():
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
 
 
async def save_upload_file(upload_file: UploadFile) -> str:
    ensure_upload_dir()
 
    ext = ""
    if upload_file.filename and "." in upload_file.filename:
        ext = "." + upload_file.filename.split(".")[-1].lower()
 
    file_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(settings.upload_dir, file_name)
 
    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)
 
    return file_path
 
 
def delete_file_safely(file_path: str):
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

