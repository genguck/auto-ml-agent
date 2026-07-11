import os
import re
import time
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Callable
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyType(Enum):
    OVERFITTING = "overfitting"
    LOSS_EXPLOSION = "loss_explosion"
    NAN_LOSS = "nan_loss"
    STAGNATION = "stagnation"


@dataclass
class EpochMetric:
    epoch: int
    train_loss: float
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    timestamp: float = 0.0


@dataclass
class AnomalyRecord:
    type: AnomalyType
    epoch: int
    message: str
    timestamp: float = 0.0


class TrainingMonitor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.anomaly_detection = self.config.get("anomaly_detection", {}).get("enabled", True)
        
        self.threshold_overfitting = self.config.get("anomaly_detection", {}).get("threshold_overfitting", 0.3)
        self.threshold_stagnation = self.config.get("anomaly_detection", {}).get("threshold_stagnation", 5)
        self.threshold_loss_explosion = self.config.get("anomaly_detection", {}).get("threshold_loss_explosion", 10.0)

        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.metrics: List[EpochMetric] = []
        self.anomalies: List[AnomalyRecord] = []
        self.output_lines: List[str] = []
        self.on_anomaly: Optional[Callable[[AnomalyRecord], None]] = None

    def start(self, process: subprocess.Popen):
        if not self.enabled:
            return

        self.process = process
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def _monitor_loop(self):
        while self.running and self.process is not None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue

                line = line.decode("utf-8", errors="replace").strip()
                self.output_lines.append(line)
                self._parse_metrics(line)

                if self.process.poll() is not None:
                    break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                break

    def _parse_metrics(self, line: str):
        epoch_match = re.search(r"\[Epoch\s+(\d+)/", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))

            train_loss_match = re.search(r"Train Loss:\s*([\d.]+)", line)
            val_loss_match = re.search(r"Val Loss:\s*([\d.]+)", line)
            train_acc_match = re.search(r"Train Acc:\s*([\d.]+)", line)
            val_acc_match = re.search(r"Val Acc:\s*([\d.]+)", line)

            metric = EpochMetric(epoch=epoch, timestamp=time.time())
            if train_loss_match:
                metric.train_loss = float(train_loss_match.group(1))
            if val_loss_match:
                metric.val_loss = float(val_loss_match.group(1))
            if train_acc_match:
                metric.train_acc = float(train_acc_match.group(1))
            if val_acc_match:
                metric.val_acc = float(val_acc_match.group(1))

            self.metrics.append(metric)
            self._detect_anomalies(metric)

    def _detect_anomalies(self, metric: EpochMetric):
        if not self.anomaly_detection:
            return

        if metric.train_loss != metric.train_loss:
            anomaly = AnomalyRecord(
                type=AnomalyType.NAN_LOSS,
                epoch=metric.epoch,
                message=f"NaN loss detected at epoch {metric.epoch}",
                timestamp=time.time(),
            )
            self._record_anomaly(anomaly)

        if metric.train_loss > self.threshold_loss_explosion:
            anomaly = AnomalyRecord(
                type=AnomalyType.LOSS_EXPLOSION,
                epoch=metric.epoch,
                message=f"Loss explosion detected: {metric.train_loss:.4f}",
                timestamp=time.time(),
            )
            self._record_anomaly(anomaly)

        if metric.val_acc is not None and metric.train_acc is not None:
            gap = metric.train_acc - metric.val_acc
            if gap > self.threshold_overfitting:
                anomaly = AnomalyRecord(
                    type=AnomalyType.OVERFITTING,
                    epoch=metric.epoch,
                    message=f"Overfitting detected: train_acc={metric.train_acc:.4f}, val_acc={metric.val_acc:.4f}, gap={gap:.4f}",
                    timestamp=time.time(),
                )
                self._record_anomaly(anomaly)

        if len(self.metrics) >= self.threshold_stagnation + 1:
            recent_metrics = self.metrics[-self.threshold_stagnation:]
            if all(
                abs(m.train_loss - self.metrics[-self.threshold_stagnation - 1].train_loss) < 0.001
                for m in recent_metrics
            ):
                anomaly = AnomalyRecord(
                    type=AnomalyType.STAGNATION,
                    epoch=metric.epoch,
                    message=f"Training stagnation detected for {self.threshold_stagnation} epochs",
                    timestamp=time.time(),
                )
                self._record_anomaly(anomaly)

    def _record_anomaly(self, anomaly: AnomalyRecord):
        self.anomalies.append(anomaly)
        logger.warning(f"Anomaly detected: {anomaly.type.value} - {anomaly.message}")
        
        if self.on_anomaly:
            try:
                self.on_anomaly(anomaly)
            except Exception as e:
                logger.error(f"Failed to call on_anomaly callback: {e}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_status(self) -> Dict:
        status = {
            "running": self.running,
            "num_epochs": len(self.metrics),
            "num_anomalies": len(self.anomalies),
            "metrics": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_acc": m.train_acc,
                    "val_loss": m.val_loss,
                    "val_acc": m.val_acc,
                }
                for m in self.metrics
            ],
            "anomalies": [
                {
                    "type": a.type.value,
                    "epoch": a.epoch,
                    "message": a.message,
                }
                for a in self.anomalies
            ],
        }
        return status

    def get_latest_metric(self) -> Optional[EpochMetric]:
        if self.metrics:
            return self.metrics[-1]
        return None
