import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NotificationChannel(Enum):
    EMAIL = "email"
    FEISHU = "feishu"
    SERVERCHAN = "serverchan"
    BARK = "bark"
    DESKTOP = "desktop"


class NotificationManager:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.channels = self._init_channels()

    def _init_channels(self) -> Dict[str, Dict]:
        channels = {}
        config_channels = self.config.get("channels", {})
        
        for name, channel_config in config_channels.items():
            if channel_config.get("enabled", False):
                channels[name] = channel_config
        
        return channels

    def send(self, payload: Dict[str, Any]) -> Dict[str, bool]:
        if not self.enabled:
            return {}

        results = {}
        for channel_name, channel_config in self.channels.items():
            try:
                success = self._send_to_channel(channel_name, channel_config, payload)
                results[channel_name] = success
                if success:
                    logger.info(f"Notification sent successfully via {channel_name}")
                else:
                    logger.error(f"Failed to send notification via {channel_name}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name}: {e}")
                results[channel_name] = False

        return results

    def _send_to_channel(self, channel_name: str, config: Dict, payload: Dict) -> bool:
        channel = NotificationChannel(channel_name)
        
        if channel == NotificationChannel.EMAIL:
            return self._send_email(config, payload)
        elif channel == NotificationChannel.FEISHU:
            return self._send_feishu(config, payload)
        elif channel == NotificationChannel.SERVERCHAN:
            return self._send_serverchan(config, payload)
        elif channel == NotificationChannel.BARK:
            return self._send_bark(config, payload)
        elif channel == NotificationChannel.DESKTOP:
            return self._send_desktop(config, payload)
        
        return False

    def _send_email(self, config: Dict, payload: Dict) -> bool:
        smtp_server = config.get("smtp_server", "")
        smtp_port = config.get("smtp_port", 587)
        smtp_user = config.get("smtp_user", "")
        smtp_password = os.getenv(config.get("smtp_password_env", ""), "")
        to_email = config.get("to_email", "")

        if not all([smtp_server, smtp_user, smtp_password, to_email]):
            logger.error("Email configuration incomplete")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = to_email
            msg["Subject"] = payload.get("title", "AutoML Experiment Notification")

            body = payload.get("message", "")
            msg.attach(MIMEText(body, "plain"))

            attachments = payload.get("attachments", [])
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    with open(attachment_path, "rb") as f:
                        part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
                    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(attachment_path)}"'
                    msg.attach(part)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, to_email, msg.as_string())

            return True
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return False

    def _send_feishu(self, config: Dict, payload: Dict) -> bool:
        webhook_url = os.getenv(config.get("webhook_url_env", ""), "")
        
        if not webhook_url:
            logger.error("Feishu webhook URL not configured")
            return False

        try:
            title = payload.get("title", "AutoML Experiment Notification")
            message = payload.get("message", "")
            status = payload.get("status", "completed")
            
            status_colors = {
                "success": "#00b42a",
                "failed": "#f53f3f",
                "completed": "#165dff",
            }
            
            data = {
                "msg_type": "interactive",
                "card": {
                    "config": {"wide_screen_mode": True},
                    "elements": [
                        {
                            "tag": "div",
                            "text": {"content": f"**{title}**", "tag": "lark_md"},
                        },
                        {
                            "tag": "div",
                            "text": {"content": message, "tag": "lark_md"},
                        },
                        {
                            "tag": "div",
                            "fields": [
                                {
                                    "is_short": True,
                                    "text": {
                                        "content": f"<font color='{status_colors.get(status, '#86909c')}'>{status.upper()}</font>",
                                        "tag": "lark_md",
                                    },
                                },
                                {
                                    "is_short": True,
                                    "text": {
                                        "content": f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                        "tag": "lark_md",
                                    },
                                },
                            ],
                        },
                    ],
                },
            }

            response = requests.post(webhook_url, json=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Feishu send error: {e}")
            return False

    def _send_serverchan(self, config: Dict, payload: Dict) -> bool:
        send_key = os.getenv(config.get("send_key_env", ""), "")
        
        if not send_key:
            logger.error("ServerChan send key not configured")
            return False

        try:
            title = payload.get("title", "AutoML Experiment Notification")
            message = payload.get("message", "")[:2000]
            
            url = f"https://sctapi.ftqq.com/{send_key}.send"
            params = {"title": title, "desp": message}
            
            response = requests.post(url, params=params)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ServerChan send error: {e}")
            return False

    def _send_bark(self, config: Dict, payload: Dict) -> bool:
        device_key = os.getenv(config.get("device_key_env", ""), "")
        
        if not device_key:
            logger.error("Bark device key not configured")
            return False

        try:
            title = payload.get("title", "AutoML Experiment")
            message = payload.get("message", "")[:500]
            
            url = f"https://api.day.app/{device_key}/{title}/{message}"
            response = requests.get(url)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Bark send error: {e}")
            return False

    def _send_desktop(self, config: Dict, payload: Dict) -> bool:
        try:
            title = payload.get("title", "AutoML Experiment Notification")
            message = payload.get("message", "")
            
            if os.name == "nt":
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)
            elif os.name == "posix":
                os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
            
            return True
        except Exception as e:
            logger.error(f"Desktop notification error: {e}")
            return False

    def build_payload(self, experiment_summary: Dict, report_path: Optional[str] = None, zip_path: Optional[str] = None) -> Dict:
        status = experiment_summary.get("status", "completed")
        title = f"AutoML Experiment {status.capitalize()}"
        
        message = f"Experiment Summary:\n\n"
        message += f"- Status: {status}\n"
        
        if "experiment_id" in experiment_summary:
            message += f"- Experiment ID: {experiment_summary['experiment_id']}\n"
        
        if "model_name" in experiment_summary:
            message += f"- Model: {experiment_summary['model_name']}\n"
        
        if "metrics" in experiment_summary:
            metrics = experiment_summary["metrics"]
            if "best_val_acc" in metrics:
                message += f"- Best Val Accuracy: {metrics['best_val_acc']:.4f}\n"
            if "best_mAP_50_95" in metrics:
                message += f"- Best mAP@0.5:0.95: {metrics['best_mAP_50_95']:.4f}\n"
        
        if "duration" in experiment_summary:
            duration = experiment_summary["duration"]
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            message += f"- Duration: {hours}h {minutes}m {seconds}s\n"
        
        if "anomalies" in experiment_summary:
            anomalies = experiment_summary["anomalies"]
            if anomalies:
                message += f"- Anomalies detected: {len(anomalies)}\n"
        
        if report_path:
            message += f"\nReport: {report_path}\n"
        
        if zip_path:
            message += f"Package: {zip_path}\n"

        payload = {
            "title": title,
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "experiment_summary": experiment_summary,
        }

        attachments = []
        if report_path and os.path.exists(report_path):
            attachments.append(report_path)
        if zip_path and os.path.exists(zip_path):
            attachments.append(zip_path)
        
        if attachments:
            payload["attachments"] = attachments

        return payload
