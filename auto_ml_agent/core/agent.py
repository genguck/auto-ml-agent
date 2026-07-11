import os
import json
import re
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..utils.logger import get_logger
from ..utils.config import get_config_value
from ..templates.base import TemplateRegistry
from .environment import EnvironmentManager
from .executor import TrainingExecutor
from .report import ReportGenerator
from .packager import ExperimentPackager
from .notifier import NotificationManager

logger = get_logger(__name__)


class TaskType(Enum):
    SETUP_ENV = "setup_env"
    INSTALL_PACKAGES = "install_packages"
    GENERATE_CODE = "generate_code"
    RUN_TRAINING = "run_training"
    GENERATE_REPORT = "generate_report"
    PACKAGE = "package"
    NOTIFY = "notify"


class Task:
    def __init__(self, type: TaskType, config: Dict[str, Any]):
        self.type = type
        self.config = config
        self.id = f"{type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status = "pending"
        self.result: Optional[Dict] = None

    def __repr__(self):
        return f"Task(id={self.id}, type={self.type.value}, status={self.status})"


class ExperimentAgent:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.environment = EnvironmentManager(self.config.get("environment", {}))
        self.executor = TrainingExecutor(self.config.get("executor", {}))
        self.report_generator = ReportGenerator(self.config.get("output", {}))
        self.packager = ExperimentPackager()
        self.notifier = NotificationManager(self.config.get("notification", {}))
        self.tasks: List[Task] = []

    def parse_natural_language(self, query: str) -> Dict[str, Any]:
        logger.info(f"Parsing natural language query: {query}")
        
        experiment_type = "ImageClassification"
        model_name = "ResNet50"
        num_classes = 10
        epochs = 40
        batch_size = 32
        learning_rate = 0.001
        data_dir = "./data"

        if "text" in query.lower() or "nlp" in query.lower() or "bert" in query.lower():
            experiment_type = "TextClassification"
            model_name = "bert-base-chinese"
            epochs = 10
            learning_rate = 2e-5

        if "object" in query.lower() or "detection" in query.lower():
            experiment_type = "ObjectDetection"
            model_name = "FasterRCNN_ResNet50_FPN"
            epochs = 20
            batch_size = 8
            learning_rate = 0.005

        resnet_match = re.search(r"resnet(\d+)", query.lower())
        if resnet_match:
            model_name = f"ResNet{resnet_match.group(1)}"

        vgg_match = re.search(r"vgg(\d+)", query.lower())
        if vgg_match:
            model_name = f"VGG{vgg_match.group(1)}"

        epoch_match = re.search(r"(\d+)\s*epoch", query.lower())
        if epoch_match:
            epochs = int(epoch_match.group(1))

        class_match = re.search(r"(\d+)\s*class", query.lower())
        if class_match:
            num_classes = int(class_match.group(1))

        batch_match = re.search(r"batch\s*size\s*(\d+)", query.lower())
        if batch_match:
            batch_size = int(batch_match.group(1))

        lr_match = re.search(r"lr\s*=\s*([\d.e-]+)", query.lower())
        if lr_match:
            learning_rate = float(lr_match.group(1))

        data_match = re.search(r"data\s*[=:]\s*([^\s]+)", query.lower())
        if data_match:
            data_dir = data_match.group(1)

        result = {
            "experiment_type": experiment_type,
            "model_name": model_name,
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "data_dir": data_dir,
        }

        logger.info(f"Parsed configuration: {result}")
        return result

    def plan_tasks(self, config: Dict[str, Any]) -> List[Task]:
        logger.info("Planning tasks")
        
        tasks = []
        
        tasks.append(Task(
            type=TaskType.SETUP_ENV,
            config={"conda_path": self.config.get("environment", {}).get("conda_path", "auto_ml_env")}
        ))

        tasks.append(Task(
            type=TaskType.INSTALL_PACKAGES,
            config={"packages": self.config.get("environment", {}).get("packages", [])}
        ))

        tasks.append(Task(
            type=TaskType.GENERATE_CODE,
            config=config
        ))

        tasks.append(Task(
            type=TaskType.RUN_TRAINING,
            config=config
        ))

        tasks.append(Task(
            type=TaskType.GENERATE_REPORT,
            config={"output_dir": config.get("output_dir", "./outputs")}
        ))

        tasks.append(Task(
            type=TaskType.PACKAGE,
            config={"source_dir": config.get("output_dir", "./outputs")}
        ))

        tasks.append(Task(
            type=TaskType.NOTIFY,
            config={}
        ))

        self.tasks = tasks
        logger.info(f"Planned {len(tasks)} tasks")
        return tasks

    def execute_task(self, task: Task) -> Dict:
        logger.info(f"Executing task: {task.type.value}")
        task.status = "running"

        try:
            if task.type == TaskType.SETUP_ENV:
                result = self._handle_setup_env(task.config)
            elif task.type == TaskType.INSTALL_PACKAGES:
                result = self._handle_install_packages(task.config)
            elif task.type == TaskType.GENERATE_CODE:
                result = self._handle_generate_code(task.config)
            elif task.type == TaskType.RUN_TRAINING:
                result = self._handle_run_training(task.config)
            elif task.type == TaskType.GENERATE_REPORT:
                result = self._handle_generate_report(task.config)
            elif task.type == TaskType.PACKAGE:
                result = self._handle_package(task.config)
            elif task.type == TaskType.NOTIFY:
                result = self._handle_notify(task.config)
            else:
                result = {"status": "error", "error": f"Unknown task type: {task.type}"}

            task.status = "completed" if result.get("status") in ["success", "completed"] else "failed"
            task.result = result
            logger.info(f"Task {task.type.value} completed with status: {task.status}")
            return result

        except Exception as e:
            task.status = "failed"
            task.result = {"status": "error", "error": str(e)}
            logger.error(f"Task {task.type.value} failed: {e}")
            return task.result

    def _handle_setup_env(self, config: Dict) -> Dict:
        logger.info("Setting up environment")
        success = self.environment.create_conda_env()
        return {"status": "success" if success else "failed", "message": "Environment setup"}

    def _handle_install_packages(self, config: Dict) -> Dict:
        logger.info("Installing packages")
        packages = config.get("packages", [])
        success = self.environment.install_packages(packages)
        return {"status": "success" if success else "failed", "message": f"Installed {len(packages)} packages"}

    def _handle_generate_code(self, config: Dict) -> Dict:
        logger.info("Generating code")
        experiment_type = config.get("experiment_type", "ImageClassification")
        
        try:
            script_content = self.executor.generate_script(experiment_type, config)
            output_dir = config.get("output_dir", "./outputs")
            script_path = self.executor.save_script(script_content, output_dir)
            return {"status": "success", "script_path": script_path, "message": "Code generated"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _handle_run_training(self, config: Dict) -> Dict:
        logger.info("Running training")
        experiment_type = config.get("experiment_type", "ImageClassification")
        
        try:
            result = self.executor.execute(experiment_type, config)
            
            if result.get("status") == "success":
                self.last_training_result = result
            return result
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _handle_generate_report(self, config: Dict) -> Dict:
        logger.info("Generating report")
        output_dir = config.get("output_dir", "./outputs")
        
        try:
            results = self.report_generator.collect_results(output_dir)
            report_path = self.report_generator.generate(results, output_dir)
            self.report_generator.generate_plots(results, output_dir)
            
            self.last_report_path = report_path
            return {"status": "success", "report_path": report_path, "message": "Report generated"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _handle_package(self, config: Dict) -> Dict:
        logger.info("Packaging experiment")
        source_dir = config.get("source_dir", "./outputs")
        output_dir = config.get("output_dir", "./outputs")
        
        try:
            zip_path = self.packager.package(source_dir, output_dir)
            
            self.last_zip_path = zip_path
            return {"status": "success", "zip_path": zip_path, "message": "Package created"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _handle_notify(self, config: Dict) -> Dict:
        logger.info("Sending notifications")
        
        experiment_summary = {}
        if hasattr(self, 'last_training_result') and self.last_training_result:
            experiment_summary = {
                "status": self.last_training_result.get("status", "completed"),
                "metrics": self.last_training_result.get("results", {}),
            }
            if "results" in self.last_training_result:
                experiment_summary["model_name"] = self.last_training_result["results"].get("model", "")

        if hasattr(self, 'last_report_path'):
            experiment_summary["report_path"] = self.last_report_path

        if hasattr(self, 'last_zip_path'):
            experiment_summary["zip_path"] = self.last_zip_path

        payload = self.notifier.build_payload(experiment_summary)
        results = self.notifier.send(payload)
        
        return {"status": "completed", "channels": results, "message": "Notifications sent"}

    def run(self, query: Optional[str] = None, config: Optional[Dict] = None) -> Dict:
        if query and not config:
            config = self.parse_natural_language(query)
        
        if not config:
            return {"status": "error", "error": "Either query or config must be provided"}

        tasks = self.plan_tasks(config)
        results = []

        for task in tasks:
            result = self.execute_task(task)
            results.append({
                "task_type": task.type.value,
                "status": task.status,
                "result": result,
            })

            if task.status == "failed" and task.type not in [TaskType.NOTIFY]:
                logger.warning(f"Task {task.type.value} failed, continuing...")

        return {
            "status": "completed",
            "tasks": results,
            "summary": {
                "total_tasks": len(tasks),
                "completed_tasks": sum(1 for t in tasks if t.status == "completed"),
                "failed_tasks": sum(1 for t in tasks if t.status == "failed"),
            },
        }

    def get_task_status(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_all_task_statuses(self) -> List[Dict]:
        return [
            {
                "id": task.id,
                "type": task.type.value,
                "status": task.status,
                "result": task.result,
            }
            for task in self.tasks
        ]
