import os
import subprocess
import json
from typing import List, Optional, Dict
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnvironmentManager:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.conda_path = self.config.get("conda_path", "auto_ml_env")
        self.python_version = self.config.get("python_version", "3.10")
        self.packages = self.config.get("packages", [])

    def check_conda(self) -> bool:
        try:
            result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def check_gpu(self) -> Dict:
        gpu_info = {
            "available": False,
            "devices": [],
            "driver_version": "",
        }

        try:
            import torch

            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["count"] = torch.cuda.device_count()
                gpu_info["devices"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                gpu_info["driver_version"] = torch.version.cuda or ""
        except ImportError:
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_info["available"] = True
                for line in result.stdout.split("\n"):
                    if "Driver Version" in line:
                        gpu_info["driver_version"] = line.strip()
                        break
        except FileNotFoundError:
            pass

        return gpu_info

    def create_conda_env(self) -> bool:
        if self.check_conda():
            logger.info(f"Creating conda environment: {self.conda_path}")
            cmd = f"conda create -n {self.conda_path} python={self.python_version} -y"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Conda environment created successfully")
                return True
            else:
                logger.error(f"Failed to create conda environment: {result.stderr}")
                return False
        else:
            logger.warning("Conda not found, skipping environment creation")
            return False

    def install_packages(self, packages: Optional[List[str]] = None) -> bool:
        packages_to_install = packages or self.packages
        if not packages_to_install:
            return True

        if self.check_conda():
            logger.info(f"Installing packages: {packages_to_install}")
            packages_str = " ".join(packages_to_install)
            cmd = f"conda run -n {self.conda_path} pip install {packages_str}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Packages installed successfully")
                return True
            else:
                logger.error(f"Failed to install packages: {result.stderr}")
                return False
        else:
            logger.warning("Conda not found, trying pip install in current environment")
            packages_str = " ".join(packages_to_install)
            cmd = f"pip install {packages_str}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0

    def install_pytorch(self, use_gpu: bool = True) -> bool:
        logger.info(f"Installing PyTorch (GPU: {use_gpu})")
        if use_gpu:
            cmd = (
                "conda run -n auto_ml_env pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu118"
            )
        else:
            cmd = (
                "conda run -n auto_ml_env pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("PyTorch installed successfully")
            return True
        else:
            logger.warning(f"Failed to install PyTorch via conda, trying pip directly: {result.stderr}")
            if use_gpu:
                cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            else:
                cmd = "pip install torch torchvision torchaudio"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0

    def get_env_path(self) -> str:
        if self.check_conda():
            result = subprocess.run(
                ["conda", "info", "--json"], capture_output=True, text=True
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                envs_dirs = info.get("envs_dirs", [])
                for env_dir in envs_dirs:
                    env_path = os.path.join(env_dir, self.conda_path)
                    if os.path.exists(env_path):
                        return env_path
        return ""

    def activate_env(self) -> str:
        if self.check_conda():
            return f"conda activate {self.conda_path}"
        return ""

    def run_in_env(self, command: str) -> subprocess.CompletedProcess:
        if self.check_conda():
            cmd = f"conda run -n {self.conda_path} {command}"
        else:
            cmd = command
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
