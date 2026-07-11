import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from ..utils.io import write_file, ensure_dir, list_files
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    experiment_id: str
    experiment_type: str
    model_name: str
    status: str
    metrics: Dict[str, Any]
    timestamp: str
    output_dir: str
    duration: Optional[float] = None
    anomalies: List[Dict] = None


class ReportGenerator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.output_base_dir = self.config.get("output_base_dir", "./outputs")

    def collect_results(self, output_dir: str) -> List[ExperimentResult]:
        results = []
        result_files = list_files(output_dir, ".json")
        
        for result_file in result_files:
            if "results.json" in result_file:
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    experiment_id = os.path.basename(os.path.dirname(result_file))
                    experiment_type = self._detect_experiment_type(result_file)
                    
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        experiment_type=experiment_type,
                        model_name=data.get("model", ""),
                        status="success",
                        metrics=data,
                        timestamp=data.get("timestamp", datetime.now().isoformat()),
                        output_dir=os.path.dirname(result_file),
                        anomalies=data.get("anomalies", []),
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to parse result file {result_file}: {e}")
        
        return results

    def _detect_experiment_type(self, file_path: str) -> str:
        dir_name = os.path.dirname(file_path)
        if "image" in dir_name.lower():
            return "ImageClassification"
        elif "text" in dir_name.lower():
            return "TextClassification"
        elif "object" in dir_name.lower() or "detection" in dir_name.lower():
            return "ObjectDetection"
        return "Unknown"

    def generate_plots(self, results: List[ExperimentResult], output_dir: str):
        ensure_dir(output_dir)
        
        if not results:
            return

        for result in results:
            metrics = result.metrics
            train_losses = metrics.get("train_losses", [])
            val_losses = metrics.get("val_losses", [])
            train_accs = metrics.get("train_accs", [])
            val_accs = metrics.get("val_accs", [])
            
            if train_losses:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label="Train Loss")
                if val_losses:
                    plt.plot(val_losses, label="Val Loss")
                plt.title(f"{result.model_name} - Loss Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                
                plt.subplot(1, 2, 2)
                if train_accs:
                    plt.plot(train_accs, label="Train Acc")
                if val_accs:
                    plt.plot(val_accs, label="Val Acc")
                plt.title(f"{result.model_name} - Accuracy Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                
                plot_path = os.path.join(output_dir, f"{result.experiment_id}_curves.png")
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Plot saved to: {plot_path}")

    def generate(self, results: List[ExperimentResult], output_dir: str) -> str:
        ensure_dir(output_dir)
        
        report = "# AutoML Experiment Report\n\n"
        report += f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Total experiments:** {len(results)}\n\n"

        report += "---\n\n"
        report += "## 1. Experiment Overview\n\n"
        
        if results:
            report += "| ID | Type | Model | Status | Best Metric |\n"
            report += "|----|------|-------|--------|-------------|\n"
            
            for result in results:
                best_metric = self._get_best_metric(result)
                report += f"| {result.experiment_id} | {result.experiment_type} | {result.model_name} | {result.status} | {best_metric} |\n"
        else:
            report += "No experiment results found.\n"

        report += "\n---\n\n"
        report += "## 2. Experiment Settings\n\n"
        
        for i, result in enumerate(results, 1):
            report += f"### Experiment {i}: {result.experiment_id}\n\n"
            report += f"- **Type:** {result.experiment_type}\n"
            report += f"- **Model:** {result.model_name}\n"
            report += f"- **Epochs:** {result.metrics.get('epochs', 'N/A')}\n"
            report += f"- **Output Directory:** {result.output_dir}\n"
            report += f"- **Timestamp:** {result.timestamp}\n\n"

        report += "---\n\n"
        report += "## 3. Experiment Results\n\n"
        
        for i, result in enumerate(results, 1):
            report += f"### Experiment {i}: {result.model_name}\n\n"
            metrics = result.metrics
            
            if "best_val_acc" in metrics:
                report += f"- **Best Validation Accuracy:** {metrics['best_val_acc']:.4f}\n"
                report += f"- **Final Train Accuracy:** {metrics['final_train_acc']:.4f}\n"
                report += f"- **Final Validation Accuracy:** {metrics['final_val_acc']:.4f}\n"
            
            if "best_mAP_50_95" in metrics:
                report += f"- **Best mAP@0.5:0.95:** {metrics['best_mAP_50_95']:.4f}\n"
            
            report += "\n"

        report += "---\n\n"
        report += "## 4. Result Analysis\n\n"
        
        if len(results) > 1:
            report += "### Comparison\n\n"
            best_result = max(results, key=self._get_best_metric_value)
            report += f"**Best performer:** {best_result.model_name} (Experiment {best_result.experiment_id})\n\n"

        for result in results:
            anomalies = result.anomalies or []
            if anomalies:
                report += f"### Anomalies in {result.experiment_id}\n\n"
                for anomaly in anomalies:
                    report += f"- {anomaly.get('type', 'Unknown')}: {anomaly.get('message', '')}\n"
                report += "\n"

        report += "---\n\n"
        report += "## 5. Conclusion\n\n"
        
        if results:
            successful = [r for r in results if r.status == "success"]
            failed = [r for r in results if r.status != "success"]
            
            report += f"- **Successful experiments:** {len(successful)}\n"
            report += f"- **Failed experiments:** {len(failed)}\n\n"
            
            if successful:
                best_result = max(successful, key=self._get_best_metric_value)
                report += f"**Recommendation:** The best model is {best_result.model_name} with {self._get_best_metric(best_result)}.\n"
        else:
            report += "No experiments were executed.\n"

        report_path = os.path.join(output_dir, "experiment_report.md")
        write_file(report_path, report)
        logger.info(f"Report saved to: {report_path}")
        
        return report_path

    def _get_best_metric(self, result: ExperimentResult) -> str:
        metrics = result.metrics
        if "best_val_acc" in metrics:
            return f"Val Acc: {metrics['best_val_acc']:.4f}"
        elif "best_mAP_50_95" in metrics:
            return f"mAP@0.5:0.95: {metrics['best_mAP_50_95']:.4f}"
        return "N/A"

    def _get_best_metric_value(self, result: ExperimentResult) -> float:
        metrics = result.metrics
        if "best_val_acc" in metrics:
            return metrics["best_val_acc"]
        elif "best_mAP_50_95" in metrics:
            return metrics["best_mAP_50_95"]
        return 0.0

    def analyze_with_llm(self, report_path: str) -> str:
        logger.info("LLM analysis is not implemented yet.")
        return "LLM analysis is not implemented yet."
