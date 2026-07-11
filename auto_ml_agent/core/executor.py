import os
import json
import subprocess
from typing import Dict, Optional, Any
from ..templates.base import TemplateRegistry
from ..utils.io import write_file, ensure_dir
from ..utils.logger import get_logger
from .monitor import TrainingMonitor

logger = get_logger(__name__)


class TrainingExecutor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.output_base_dir = self.config.get("output_base_dir", "./outputs")
        self.monitor = TrainingMonitor(self.config.get("monitor", {}))

    def generate_script(self, experiment_type: str, config: Dict[str, Any]) -> str:
        template = TemplateRegistry.get(experiment_type)
        if not template:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        if not template.validate_config(config):
            raise ValueError(f"Invalid config for {experiment_type}")

        script_content = template.render(config)
        return script_content

    def save_script(self, script_content: str, output_dir: str, filename: str = "train.py") -> str:
        ensure_dir(output_dir)
        script_path = os.path.join(output_dir, filename)
        write_file(script_path, script_content)
        logger.info(f"Script saved to: {script_path}")
        return script_path

    def run_training(self, script_path: str, args: Optional[Dict] = None) -> Dict:
        logger.info(f"Starting training: {script_path}")
        
        args_str = ""
        if args:
            for key, value in args.items():
                args_str += f" --{key} {value}"

        cmd = f"python {script_path}{args_str}"
        logger.info(f"Command: {cmd}")

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        self.monitor.start(process)

        try:
            stdout, _ = process.communicate(timeout=7200)
        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            process.kill()
            return {"status": "timeout", "error": "Training timed out"}

        self.monitor.stop()

        if process.returncode != 0:
            logger.error(f"Training failed with exit code {process.returncode}")
            return {
                "status": "failed",
                "exit_code": process.returncode,
                "error": stdout[-2000:] if len(stdout) > 2000 else stdout,
            }

        return {
            "status": "success",
            "exit_code": process.returncode,
            "stdout": stdout,
            "monitor_data": self.monitor.get_status(),
        }

    def parse_results(self, output_dir: str) -> Optional[Dict]:
        results_path = os.path.join(output_dir, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to parse results: {e}")
        return None

    def execute(self, experiment_type: str, config: Dict[str, Any]) -> Dict:
        output_dir = config.get("output_dir", self.output_base_dir)
        ensure_dir(output_dir)

        try:
            script_content = self.generate_script(experiment_type, config)
            script_path = self.save_script(script_content, output_dir)
            
            args = {
                "data_dir": config.get("data_dir", "./data"),
                "model": config.get("model_name", "ResNet50"),
                "num_classes": config.get("num_classes", 10),
                "batch_size": config.get("batch_size", 32),
                "epochs": config.get("epochs", 40),
                "lr": config.get("learning_rate", 0.001),
                "output_dir": output_dir,
            }

            if experiment_type == "TextClassification":
                args["data_path"] = config.get("data_path", "./data.csv")
                args["model_name"] = config.get("model_name", "bert-base-chinese")
                args["max_len"] = config.get("max_len", 128)
            elif experiment_type == "ObjectDetection":
                args["train_annot"] = config.get("train_annot", "./annotations/train.json")
                args["val_annot"] = config.get("val_annot", "./annotations/val.json")

            result = self.run_training(script_path, args)
            
            if result["status"] == "success":
                parsed_results = self.parse_results(output_dir)
                if parsed_results:
                    result["results"] = parsed_results

            return result
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {"status": "error", "error": str(e)}
