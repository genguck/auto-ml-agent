import os
import zipfile
from datetime import datetime
from typing import List, Optional
from ..utils.io import list_files, ensure_dir
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentPackager:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.exclude_patterns = [
            ".pyc",
            "__pycache__",
            ".tmp",
            ".temp",
            ".git",
            ".gitignore",
            ".DS_Store",
            ".idea",
            ".vscode",
            "*.egg-info",
            "*.log",
        ]

    def should_exclude(self, path: str) -> bool:
        path_lower = path.lower()
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if path_lower.endswith(pattern[1:]):
                    return True
            else:
                if pattern in path_lower:
                    return True
        return False

    def package(self, source_dir: str, output_dir: str, include_files: Optional[List[str]] = None) -> str:
        ensure_dir(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"experiment_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        logger.info(f"Packaging {source_dir} to {zip_path}")

        all_files = []
        if include_files:
            for f in include_files:
                if os.path.exists(f):
                    all_files.append(f)
        else:
            all_files = list_files(source_dir)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in all_files:
                if self.should_exclude(file_path):
                    continue
                
                rel_path = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, rel_path)
                logger.debug(f"Added: {rel_path}")

        logger.info(f"Package created: {zip_path} ({os.path.getsize(zip_path)} bytes)")
        return zip_path

    def package_multiple(self, dirs: List[str], output_dir: str) -> str:
        ensure_dir(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"experiments_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        logger.info(f"Packaging {len(dirs)} directories to {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for source_dir in dirs:
                if not os.path.exists(source_dir):
                    logger.warning(f"Directory not found: {source_dir}")
                    continue
                
                for file_path in list_files(source_dir):
                    if self.should_exclude(file_path):
                        continue
                    
                    rel_path = os.path.relpath(file_path, os.path.dirname(source_dir))
                    zipf.write(file_path, rel_path)
                    logger.debug(f"Added: {rel_path}")

        logger.info(f"Package created: {zip_path} ({os.path.getsize(zip_path)} bytes)")
        return zip_path
