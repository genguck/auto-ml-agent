import os
import subprocess
from typing import List, Optional, Tuple


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def list_files(dir_path: str, pattern: Optional[str] = None) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for filename in filenames:
            if pattern and not filename.endswith(pattern):
                continue
            files.append(os.path.join(root, filename))
    return files


def ensure_dir(dir_path: str) -> None:
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def run_command(
    cmd: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    capture_output: bool = True,
) -> Tuple[int, str, str]:
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout.strip(), stderr.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return -1, "", "Command timed out"


def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


def get_dir_size(dir_path: str) -> int:
    total_size = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    return total_size


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def remove_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def remove_dir(dir_path: str) -> None:
    import shutil

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
