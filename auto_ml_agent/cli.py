import click
import json
import os
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .utils.config import load_config, get_config_value
from .utils.logger import get_logger
from .templates.base import TemplateRegistry
from .core.agent import ExperimentAgent
from .core.environment import EnvironmentManager
from .core.executor import TrainingExecutor
from .core.report import ReportGenerator
from .core.packager import ExperimentPackager
from .core.notifier import NotificationManager

logger = get_logger(__name__)
console = Console()


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = {"config": load_config()}


@cli.command(help="Run experiment with natural language query")
@click.argument("query")
@click.pass_context
def run(ctx, query):
    console.print(f"[bold blue]Running experiment:[/bold blue] {query}")
    
    agent = ExperimentAgent(ctx.obj["config"])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Processing query and planning tasks...", total=1)
        
        try:
            config = agent.parse_natural_language(query)
            progress.advance(task)
            
            console.print(f"\n[bold green]Parsed configuration:[/bold green]")
            for key, value in config.items():
                console.print(f"  {key}: {value}")
            
            result = agent.run(config=config)
            
            console.print("\n[bold green]Experiment completed![/bold green]")
            print_task_summary(result)
            
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            logger.error(f"Experiment failed: {e}")


@cli.command(help="Quick run with specified parameters")
@click.option("--type", default="ImageClassification", help="Experiment type")
@click.option("--model", default="ResNet50", help="Model name")
@click.option("--data", default="./data", help="Data directory")
@click.option("--epochs", default=40, type=int, help="Number of epochs")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--num-classes", default=10, type=int, help="Number of classes")
@click.pass_context
def quick(ctx, type, model, data, epochs, batch_size, num_classes):
    console.print(f"[bold blue]Quick run:[/bold blue] {type} with {model}")
    
    config = {
        "experiment_type": type,
        "model_name": model,
        "data_dir": data,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "learning_rate": 0.001,
        "output_dir": "./outputs",
    }
    
    if type == "TextClassification":
        config["learning_rate"] = 2e-5
        config["epochs"] = 10
        
    agent = ExperimentAgent(ctx.obj["config"])
    
    try:
        result = agent.run(config=config)
        console.print("\n[bold green]Quick run completed![/bold green]")
        print_task_summary(result)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")


@cli.command(help="List available experiment templates")
def list_templates():
    templates = TemplateRegistry.get_all()
    
    console.print("[bold blue]Available Experiment Templates:[/bold blue]")
    console.print("")
    
    for name, template in templates.items():
        default_config = template.get_default_config()
        
        panel = Panel(
            f"[bold]{name}[/bold]\n\n"
            f"[green]Default Configuration:[/green]\n"
            + "\n".join([f"  {key}: {value}" for key, value in default_config.items()]),
            expand=False,
        )
        console.print(panel)
        console.print("")


@cli.command(help="Check GPU status")
def gpu():
    env_manager = EnvironmentManager()
    gpu_info = env_manager.check_gpu()
    
    if gpu_info["available"]:
        console.print("[bold green]GPU is available![/bold green]")
        console.print(f"  Count: {gpu_info.get('count', 0)}")
        console.print(f"  Devices: {', '.join(gpu_info.get('devices', []))}")
        console.print(f"  Driver: {gpu_info.get('driver_version', '')}")
    else:
        console.print("[bold yellow]No GPU detected, using CPU mode[/bold yellow]")


@cli.command(help="Setup environment")
@click.option("--install-pytorch", is_flag=True, help="Install PyTorch")
@click.pass_context
def setup(ctx, install_pytorch):
    console.print("[bold blue]Setting up environment...[/bold blue]")
    
    env_manager = EnvironmentManager(ctx.obj["config"].get("environment", {}))
    
    if env_manager.check_conda():
        console.print("[green]Conda is available[/green]")
        success = env_manager.create_conda_env()
        if success:
            console.print("[green]✓ Conda environment created[/green]")
        else:
            console.print("[yellow]Conda environment may already exist[/yellow]")
        
        packages = ctx.obj["config"].get("environment", {}).get("packages", [])
        if packages:
            console.print(f"Installing {len(packages)} packages...")
            env_manager.install_packages(packages)
            console.print("[green]✓ Packages installed[/green]")
        
        if install_pytorch:
            console.print("Installing PyTorch...")
            gpu_info = env_manager.check_gpu()
            success = env_manager.install_pytorch(use_gpu=gpu_info["available"])
            if success:
                console.print("[green]✓ PyTorch installed[/green]")
            else:
                console.print("[red]Failed to install PyTorch[/red]")
    else:
        console.print("[yellow]Conda not found, skipping environment setup[/yellow]")


@cli.command(help="Generate experiment report")
@click.option("--output-dir", default="./outputs", help="Output directory")
def report(output_dir):
    console.print(f"[bold blue]Generating report for:[/bold blue] {output_dir}")
    
    report_generator = ReportGenerator()
    
    try:
        results = report_generator.collect_results(output_dir)
        
        if not results:
            console.print("[yellow]No experiment results found[/yellow]")
            return
        
        report_path = report_generator.generate(results, output_dir)
        report_generator.generate_plots(results, output_dir)
        
        console.print(f"[green]✓ Report generated:[/green] {report_path}")
        console.print(f"[green]✓ Plots generated[/green]")
    except Exception as e:
        console.print(f"[red]Error generating report:[/red] {e}")


@cli.command(help="Package experiment outputs")
@click.option("--source-dir", default="./outputs", help="Source directory")
@click.option("--output-dir", default="./outputs", help="Output directory")
def package(source_dir, output_dir):
    console.print(f"[bold blue]Packaging:[/bold blue] {source_dir}")
    
    packager = ExperimentPackager()
    
    try:
        zip_path = packager.package(source_dir, output_dir)
        console.print(f"[green]✓ Package created:[/green] {zip_path}")
    except Exception as e:
        console.print(f"[red]Error packaging:[/red] {e}")


@cli.command(help="Send notification")
@click.option("--title", default="AutoML Notification", help="Notification title")
@click.option("--message", default="Experiment completed", help="Notification message")
@click.pass_context
def notify(ctx, title, message):
    console.print("[bold blue]Sending notification...[/bold blue]")
    
    notifier = NotificationManager(ctx.obj["config"].get("notification", {}))
    
    payload = {
        "title": title,
        "message": message,
        "status": "completed",
    }
    
    results = notifier.send(payload)
    
    console.print("[bold green]Notification results:[/bold green]")
    for channel, success in results.items():
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(f"  {channel}: {status}")


def print_task_summary(result):
    tasks = result.get("tasks", [])
    summary = result.get("summary", {})
    
    table = Table(title="Task Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="magenta")
    
    for task in tasks:
        status = "✅ Completed" if task["status"] == "completed" else "❌ Failed"
        table.add_row(task["task_type"], status)
    
    console.print(table)
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total: {summary.get('total_tasks', 0)}")
    console.print(f"  Completed: {summary.get('completed_tasks', 0)}")
    console.print(f"  Failed: {summary.get('failed_tasks', 0)}")


if __name__ == "__main__":
    cli()
