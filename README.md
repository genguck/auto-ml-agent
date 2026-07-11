# AutoML Agent

AI-powered autonomous ML experiment agent. Describe your experiment in natural language, and the agent automatically handles environment setup, code generation, model training, report writing, packaging, and user notification via email/Feishu/WeChat.

## Features

- **Natural Language Interface**: Describe your experiment in plain English/Chinese
- **Multiple Experiment Types**: Image Classification, Text Classification, Object Detection
- **Auto Environment Setup**: Conda environment management with GPU support
- **Real-time Monitoring**: Training progress tracking with anomaly detection
- **Automatic Report Generation**: Markdown reports with matplotlib charts
- **Packaging**: Zip packaging of all experiment artifacts
- **Multi-channel Notification**: Email, Feishu, ServerChan, Bark, Desktop

## Installation

```bash
git clone https://github.com/your-username/auto-ml-agent.git
cd auto-ml-agent
pip install -e .
```

## Usage

### Natural Language Mode

```bash
auto-ml run "用 ResNet50 在 CIFAR-10 上做图像分类，训练 40 个 epoch"
auto-ml run "使用 BERT 做文本分类，训练 10 个 epoch"
auto-ml run "用 Faster R-CNN 做目标检测，batch size 8"
```

### Quick Mode

```bash
auto-ml quick --type ImageClassification --model ResNet50 --data ./data --epochs 40
auto-ml quick --type TextClassification --model bert-base-chinese --data ./data.csv
auto-ml quick --type ObjectDetection --data ./data --epochs 20
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `auto-ml run <query>` | Run experiment with natural language query |
| `auto-ml quick` | Quick run with specified parameters |
| `auto-ml list-templates` | List available experiment templates |
| `auto-ml gpu` | Check GPU status |
| `auto-ml setup` | Setup environment |
| `auto-ml report` | Generate experiment report |
| `auto-ml package` | Package experiment outputs |
| `auto-ml notify` | Send notification |

## Configuration

Edit `config.yaml` to configure the agent:

```yaml
agent:
  llm_provider: anthropic
  model: claude-3-sonnet-20240229

environment:
  conda_path: auto_ml_env
  python_version: "3.10"

notification:
  enabled: true
  channels:
    desktop:
      enabled: true
    email:
      enabled: false
      smtp_server: smtp.example.com
      smtp_port: 587
      smtp_user: your_email@example.com
      smtp_password_env: SMTP_PASSWORD
      to_email: recipient@example.com
    feishu:
      enabled: false
      webhook_url_env: FEISHU_WEBHOOK_URL
```

## Project Structure

```
auto-ml-agent/
├── auto_ml_agent/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry (7 commands)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py              # Agent engine (task planning + execution)
│   │   ├── environment.py        # Conda environment + GPU management
│   │   ├── executor.py           # Training script generation + execution
│   │   ├── monitor.py            # Real-time monitoring + anomaly detection
│   │   ├── report.py             # Structured experiment report + charts
│   │   ├── packager.py           # Zip packaging
│   │   └── notifier.py           # Multi-channel notification
│   ├── templates/
│   │   ├── __init__.py
│   │   ├── base.py               # Template base class + registry
│   │   ├── image_classification.py   # ResNet/VGG/ViT templates
│   │   ├── text_classification.py    # BERT/RoBERTa/DistilBERT templates
│   │   └── object_detection.py       # Faster R-CNN template
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # YAML config + env vars
│       ├── io.py                 # File/command utilities
│       └── logger.py             # Rich logging
├── config.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

## Supported Models

### Image Classification
- ResNet18, ResNet50
- VGG16
- ViT-Base-16

### Text Classification
- bert-base-chinese
- roberta-base
- distilbert-base-uncased

### Object Detection
- Faster R-CNN with ResNet50 FPN

## Notification Channels

| Channel | Description |
|---------|-------------|
| **email** | SMTP email with attachments |
| **feishu** | Feishu/Lark Webhook bot |
| **serverchan** | ServerChan (WeChat push) |
| **bark** | Bark (iOS push) |
| **desktop** | System desktop notification |

## License

MIT License
