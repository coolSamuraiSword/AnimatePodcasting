#!/usr/bin/env python
"""
AnimatePodcasting CLI - A beautiful command-line interface for AnimatePodcasting

This script provides a rich, interactive command-line interface for the AnimatePodcasting
project, with progress bars, colorful output, and detailed status information.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import print as rprint
from dotenv import load_dotenv

# Add the current directory to the path so we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import project modules
try:
    from src.transcriber import WhisperTranscriber
    from src.analyzer import ContentAnalyzer
    from src.generator import ImageGenerator
    from src.project_notion import ProjectNotionManager
    from src.cache_manager import CacheManager
    from src.performance import PerformanceTracker, track, track_operation
    from src.error_handler import ErrorHandler, handle_errors, error_context, ErrorType, ErrorSeverity, RecoveryStrategy
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("animate")

# Create typer app
app = typer.Typer(
    help="AnimatePodcasting - Create animated videos from podcast audio",
    add_completion=False
)

def get_available_whisper_models() -> List[str]:
    """Get a list of available Whisper models."""
    return ["tiny", "base", "small", "medium", "large"]

def validate_audio_file(file_path: str) -> str:
    """Validate that the audio file exists."""
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/] Audio file not found: {file_path}")
        raise typer.Exit(code=1)
    return file_path

def display_cache_stats(cache_manager: CacheManager):
    """Display cache statistics in a table."""
    stats = cache_manager.get_cache_stats()
    
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)

def display_transcription_sample(transcription: Dict[str, Any]):
    """Display a sample of the transcription."""
    if not transcription or "text" not in transcription:
        return
    
    text = transcription["text"]
    sample = text[:500] + "..." if len(text) > 500 else text
    
    panel = Panel(
        sample,
        title="Transcription Sample",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)

def display_key_segments(segments: List[str], max_display: int = 3):
    """Display key segments in a nicely formatted way."""
    table = Table(title="Key Segments")
    table.add_column("Segment", style="cyan")
    
    for i, segment in enumerate(segments[:max_display]):
        # Truncate long segments
        if len(segment) > 100:
            segment = segment[:97] + "..."
        table.add_row(segment)
    
    if len(segments) > max_display:
        table.add_row(f"... and {len(segments) - max_display} more segments")
    
    console.print(table)

@app.command()
def transcribe(
    audio: str = typer.Argument(..., help="Path to the audio file", callback=validate_audio_file),
    output: str = typer.Option("outputs/transcript.json", "--output", "-o", help="Path to save the transcription JSON"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model size to use"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use cache for transcription"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (e.g., 'en' for English)"),
    show_stats: bool = typer.Option(False, "--stats", help="Show cache statistics after transcription")
):
    """Transcribe an audio file using Whisper."""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with console.status(f"[bold blue]Loading Whisper {model} model...", spinner="dots") as status:
        transcriber = WhisperTranscriber(model_size=model, use_cache=use_cache)
        
        # Update status
        status.update(f"[bold blue]Transcribing {audio}...")
        
        # Perform transcription
        transcription = transcriber.transcribe(audio, language=language)
        
        # Get duration from transcription if available
        duration = "unknown"
        if transcription.get("segments"):
            last_segment = transcription["segments"][-1]
            duration = f"{last_segment.get('end', 0):.2f} seconds"
        
        # Save transcription to file
        import json
        with open(output, "w") as f:
            json.dump(transcription, f, indent=2)
        
        # Update status with completion message
        elapsed = time.time() - start_time
        status.update(f"[bold green]Transcription completed in {elapsed:.2f}s! Saved to {output}")
        time.sleep(1)  # Give user a moment to see the completion message
    
    # Display a sample of the transcription
    display_transcription_sample(transcription)
    
    # Display some stats
    stats_table = Table(title="Transcription Stats")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Audio File", os.path.basename(audio))
    stats_table.add_row("Model", model)
    stats_table.add_row("Duration", duration)
    stats_table.add_row("Word Count", str(len(transcription.get("text", "").split())))
    stats_table.add_row("Segment Count", str(len(transcription.get("segments", []))))
    stats_table.add_row("Processing Time", f"{elapsed:.2f} seconds")
    
    console.print(stats_table)
    
    # Show cache stats if requested
    if show_stats and use_cache:
        display_cache_stats(transcriber.cache)
    
    return transcription

@app.command()
def analyze(
    transcription_file: str = typer.Argument(..., help="Path to the transcription JSON file"),
    output: str = typer.Option("outputs/key_segments.txt", "--output", "-o", help="Path to save key segments"),
    num_segments: int = typer.Option(5, "--num-segments", "-n", help="Number of key segments to extract"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use cache for analysis"),
    show_stats: bool = typer.Option(False, "--stats", help="Show cache statistics after analysis")
):
    """Extract key segments from a transcription file."""
    start_time = time.time()
    
    # Check if transcription file exists
    if not os.path.exists(transcription_file):
        console.print(f"[bold red]Error:[/] Transcription file not found: {transcription_file}")
        raise typer.Exit(code=1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load transcription from file
    import json
    with open(transcription_file, "r") as f:
        transcription = json.load(f)
    
    with console.status("[bold blue]Loading analysis model...", spinner="dots") as status:
        analyzer = ContentAnalyzer(use_cache=use_cache)
        
        # Update status
        status.update("[bold blue]Analyzing content...")
        
        # Extract key segments
        key_segments = analyzer.extract_key_segments(transcription, top_n=num_segments)
        
        # Save key segments to file
        with open(output, "w") as f:
            for segment in key_segments:
                f.write(f"{segment}\n\n")
        
        # Update status with completion message
        elapsed = time.time() - start_time
        status.update(f"[bold green]Analysis completed in {elapsed:.2f}s! Saved {len(key_segments)} key segments to {output}")
        time.sleep(1)  # Give user a moment to see the completion message
    
    # Display key segments
    display_key_segments(key_segments)
    
    # Show cache stats if requested
    if show_stats and use_cache:
        display_cache_stats(analyzer.cache)
    
    return key_segments

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    output: str = typer.Option("outputs/image.png", "--output", "-o", help="Path to save the generated image"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Animation style to use"),
    width: int = typer.Option(512, "--width", "-w", help="Image width"),
    height: int = typer.Option(512, "--height", help="Image height"),
    steps: int = typer.Option(30, "--steps", help="Number of inference steps"),
    guidance: float = typer.Option(7.5, "--guidance", "-g", help="Guidance scale")
):
    """Generate an image from a text prompt using Stable Diffusion."""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with console.status("[bold blue]Loading image generation model...", spinner="dots") as status:
        generator = ImageGenerator(animation_style=style)
        
        # Update status with prompt information
        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        status.update(f"[bold blue]Generating image for: {short_prompt}")
        
        # Generate the image
        image = generator.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance
        )
        
        # Save the image
        image.save(output)
        
        # Update status with completion message
        elapsed = time.time() - start_time
        status.update(f"[bold green]Image generated in {elapsed:.2f}s! Saved to {output}")
        time.sleep(1)  # Give user a moment to see the completion message
    
    # Display generation stats
    stats_table = Table(title="Image Generation Stats")
    stats_table.add_column("Parameter", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Prompt", short_prompt)
    stats_table.add_row("Style", style or "default")
    stats_table.add_row("Dimensions", f"{width}x{height}")
    stats_table.add_row("Inference Steps", str(steps))
    stats_table.add_row("Guidance Scale", str(guidance))
    stats_table.add_row("Output File", output)
    stats_table.add_row("Processing Time", f"{elapsed:.2f} seconds")
    
    console.print(stats_table)
    
    # Display the path where the image is saved
    console.print(f"\n[bold green]Image saved to:[/] {os.path.abspath(output)}")

@app.command()
def create(
    audio: str = typer.Argument(..., help="Path to the audio file", callback=validate_audio_file),
    output: str = typer.Option("outputs/output.mp4", "--output", "-o", help="Path to save the generated video"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model size to use"),
    num_images: int = typer.Option(5, "--num-images", "-n", help="Number of images to generate"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Animation style to use"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use cache for processing"),
    sync_notion: bool = typer.Option(False, "--notion", help="Sync with Notion"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (e.g., 'en' for English)")
):
    """Create an animated video from podcast audio (full pipeline)."""
    start_time = time.time()
    overall_start_time = start_time
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Notion if requested
    notion = None
    if sync_notion:
        with console.status("[bold blue]Connecting to Notion...", spinner="dots"):
            try:
                notion = ProjectNotionManager()
                console.print("[bold green]Connected to Notion successfully![/]")
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/] Failed to connect to Notion: {e}")
                console.print("[yellow]Continuing without Notion integration[/]")
    
    # Create a progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Add overall task
        overall_task = progress.add_task("[bold cyan]Overall Progress", total=4)
        
        # Step 1: Transcription
        transcribe_task = progress.add_task("[bold blue]Transcribing audio...", total=1)
        
        # Load the transcriber
        transcriber = WhisperTranscriber(model_size=model, use_cache=use_cache)
        
        # Start transcription
        transcription = transcriber.transcribe(audio, language=language)
        progress.update(transcribe_task, advance=1)
        progress.update(overall_task, advance=1)
        
        # Log to Notion if connected
        if notion:
            notion.log_processing_activity(
                "transcription",
                {
                    "Audio File": os.path.basename(audio),
                    "Model": model,
                    "Duration": f"{transcription.get('segments', [])[-1].get('end', 0):.2f}s",
                    "Word Count": len(transcription.get("text", "").split()),
                }
            )
        
        # Step 2: Analysis
        analysis_task = progress.add_task("[bold blue]Analyzing content...", total=1)
        
        # Load the analyzer
        analyzer = ContentAnalyzer(use_cache=use_cache)
        
        # Extract key segments
        key_segments = analyzer.extract_key_segments(transcription, top_n=num_images)
        progress.update(analysis_task, advance=1)
        progress.update(overall_task, advance=1)
        
        # Log to Notion if connected
        if notion:
            notion.log_processing_activity(
                "content_analysis",
                {
                    "Key Segments Found": len(key_segments),
                    "Sample Segment": key_segments[0][:100] + "..." if key_segments else "None"
                }
            )
        
        # Step 3: Image Generation
        image_task = progress.add_task("[bold blue]Generating images...", total=len(key_segments))
        
        # Load the image generator
        generator = ImageGenerator(animation_style=style)
        
        # Generate images
        image_paths = []
        for i, segment in enumerate(key_segments):
            progress.update(image_task, description=f"[bold blue]Generating image {i+1}/{len(key_segments)}...")
            
            # Generate image
            image_path = os.path.join(output_dir, f"frame_{i:03d}.png")
            image = generator.generate_image(segment)
            image.save(image_path)
            image_paths.append(image_path)
            
            # Log to Notion if connected
            if notion:
                notion.log_processing_activity(
                    "image_generation",
                    {
                        "Image": f"frame_{i:03d}.png",
                        "Style": style or "default",
                        "Prompt": segment[:100] + "..."
                    }
                )
            
            progress.update(image_task, advance=1)
        
        progress.update(overall_task, advance=1)
        
        # Step 4: Video Creation
        video_task = progress.add_task("[bold blue]Creating video...", total=1)
        
        # Create video
        generator.create_video(audio, image_paths, output)
        
        # Log to Notion if connected
        if notion:
            notion.log_processing_activity(
                "video_creation",
                {
                    "Output": os.path.basename(output),
                    "Images Used": len(image_paths),
                    "Audio": os.path.basename(audio)
                }
            )
            
            # Update project status
            notion.update_project_status({
                "Last Processing": "Complete",
                "Audio Processed": os.path.basename(audio),
                "Images Generated": len(image_paths),
                "Video Output": os.path.basename(output),
                "Model Used": model,
                "Animation Style": style or "default",
                "Processing Time": f"{time.time() - overall_start_time:.2f}s"
            })
        
        progress.update(video_task, advance=1)
        progress.update(overall_task, advance=1)
    
    # Display overall stats
    elapsed = time.time() - overall_start_time
    
    stats_table = Table(title="Processing Summary")
    stats_table.add_column("Stage", style="cyan")
    stats_table.add_column("Details", style="green")
    
    stats_table.add_row("Audio", os.path.basename(audio))
    stats_table.add_row("Transcript Length", f"{len(transcription.get('text', '').split())} words")
    stats_table.add_row("Key Segments", str(len(key_segments)))
    stats_table.add_row("Images Generated", str(len(image_paths)))
    stats_table.add_row("Output Video", output)
    stats_table.add_row("Total Time", f"{elapsed:.2f} seconds")
    
    console.print(stats_table)
    
    # Final success message
    console.print(f"\n[bold green]Success![/] Animated video created at: {os.path.abspath(output)}")

@app.command("cache")
def manage_cache(
    clear: bool = typer.Option(False, "--clear", help="Clear the entire cache"),
    info: bool = typer.Option(True, "--info", help="Show cache information")
):
    """Manage the cache system."""
    cache = CacheManager()
    
    if clear:
        with console.status("[bold blue]Clearing cache...", spinner="dots"):
            cache.clear_cache()
            console.print("[bold green]Cache cleared successfully![/]")
    
    if info:
        display_cache_stats(cache)

@app.command("version")
def show_version():
    """Show the version information."""
    version_table = Table(title="AnimatePodcasting", show_header=False)
    version_table.add_column(style="cyan")
    version_table.add_column(style="green")
    
    version_table.add_row("Version", "1.0.0")
    version_table.add_row("Author", "AnimatePodcasting Team")
    version_table.add_row("License", "MIT")
    version_table.add_row("Python", sys.version.split()[0])
    version_table.add_row("Platform", sys.platform)
    
    console.print(version_table)

@app.command("performance")
def manage_performance(
    report: bool = typer.Option(True, "--report", help="Show performance report"),
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save performance report to file"),
    reset: bool = typer.Option(False, "--reset", help="Reset performance metrics"),
    benchmark: bool = typer.Option(False, "--benchmark", help="Run a benchmark test suite")
):
    """Monitor and analyze application performance."""
    from src.performance import default_tracker
    
    if reset:
        with console.status("[bold blue]Resetting performance metrics...", spinner="dots"):
            default_tracker.reset()
            console.print("[bold green]Performance metrics reset![/]")
    
    if benchmark:
        # Run a simple benchmark test suite
        console.print("[bold blue]Running benchmark test suite...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[bold blue]Running benchmarks...", total=3)
            
            # Benchmark transcription
            progress.update(task, description="[bold blue]Benchmarking transcription...")
            try:
                transcriber = WhisperTranscriber(model_size="tiny")
                test_file = "data/sample_audio.mp3"
                
                if not os.path.exists(test_file):
                    # Create a dummy test file if needed
                    console.print(f"[yellow]Test file {test_file} not found. Creating dummy file...[/]")
                    os.makedirs("data", exist_ok=True)
                    with open(test_file, "w") as f:
                        f.write("Dummy test file")
                
                result = default_tracker.benchmark(
                    transcriber.transcribe,
                    test_file,
                    iterations=2  # Using fewer iterations for speed
                )
                console.print(f"[green]Transcription benchmark completed: avg={result['average_time']:.2f}s[/]")
            except Exception as e:
                console.print(f"[bold red]Error running transcription benchmark: {e}[/]")
            
            progress.update(task, advance=1)
            
            # Benchmark content analysis
            progress.update(task, description="[bold blue]Benchmarking content analysis...")
            try:
                analyzer = ContentAnalyzer()
                mock_transcription = {"text": "This is a test transcription for benchmarking", "segments": [{"text": "This is a test."}]}
                
                result = default_tracker.benchmark(
                    analyzer.extract_key_segments,
                    mock_transcription,
                    iterations=3
                )
                console.print(f"[green]Analysis benchmark completed: avg={result['average_time']:.2f}s[/]")
            except Exception as e:
                console.print(f"[bold red]Error running analysis benchmark: {e}[/]")
                
            progress.update(task, advance=1)
            
            # Benchmark image generation (simplified)
            progress.update(task, description="[bold blue]Benchmarking internal operations...")
            try:
                # Simulate an operation for benchmarking
                def test_operation():
                    time.sleep(0.1)  # Simulate work
                    return True
                
                result = default_tracker.benchmark(
                    test_operation,
                    iterations=5
                )
                console.print(f"[green]Internal benchmark completed: avg={result['average_time']:.2f}s[/]")
            except Exception as e:
                console.print(f"[bold red]Error running internal benchmark: {e}[/]")
                
            progress.update(task, advance=1)
    
    if report:
        # Generate and display performance report
        report_data = default_tracker.get_report()
        
        # Display function statistics
        if report_data["function_stats"]:
            func_table = Table(title="Function Performance")
            func_table.add_column("Function", style="cyan")
            func_table.add_column("Calls", style="blue")
            func_table.add_column("Avg Time (s)", style="green")
            func_table.add_column("Min Time (s)", style="green")
            func_table.add_column("Max Time (s)", style="green")
            func_table.add_column("Success %", style="yellow")
            
            for func_name, stats in report_data["function_stats"].items():
                func_table.add_row(
                    func_name,
                    str(stats["calls"]),
                    f"{stats['average_time']:.4f}",
                    f"{stats['min_time']:.4f}",
                    f"{stats['max_time']:.4f}",
                    f"{stats['success_rate']:.1f}%"
                )
            
            console.print(func_table)
        else:
            console.print("[yellow]No function performance data available yet.[/]")
        
        # Display operation statistics
        if report_data["operation_stats"]:
            op_table = Table(title="Operation Performance")
            op_table.add_column("Operation", style="cyan")
            op_table.add_column("Executions", style="blue")
            op_table.add_column("Avg Time (s)", style="green")
            op_table.add_column("Min Time (s)", style="green")
            op_table.add_column("Max Time (s)", style="green")
            op_table.add_column("Success %", style="yellow")
            
            for op_name, stats in report_data["operation_stats"].items():
                op_table.add_row(
                    op_name,
                    str(stats["executions"]),
                    f"{stats['average_time']:.4f}",
                    f"{stats['min_time']:.4f}",
                    f"{stats['max_time']:.4f}",
                    f"{stats['success_rate']:.1f}%"
                )
            
            console.print(op_table)
        else:
            console.print("[yellow]No operation performance data available yet.[/]")
        
        # Display benchmark summary
        if report_data["benchmark_summary"]:
            bench_table = Table(title="Benchmark Results")
            bench_table.add_column("Function", style="cyan")
            bench_table.add_column("Iterations", style="blue")
            bench_table.add_column("Avg Time (s)", style="green")
            bench_table.add_column("Min Time (s)", style="green")
            bench_table.add_column("Max Time (s)", style="green")
            
            for bench in report_data["benchmark_summary"]:
                bench_table.add_row(
                    bench["function"],
                    str(bench["iterations"]),
                    f"{bench['average_time']:.4f}",
                    f"{bench['min_time']:.4f}",
                    f"{bench['max_time']:.4f}"
                )
            
            console.print(bench_table)
    
    if save:
        with console.status(f"[bold blue]Saving performance report to {save}...", spinner="dots"):
            default_tracker.save_report(save)
            console.print(f"[bold green]Performance report saved to {save}![/]")

@app.command("errors")
def manage_errors(
    report: bool = typer.Option(True, "--report", help="Show error report"),
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save error report to file"),
    clear: bool = typer.Option(False, "--clear", help="Clear error history"),
    test: bool = typer.Option(False, "--test", help="Run a test with different error scenarios")
):
    """Manage error handling and view error reports."""
    from src.error_handler import default_handler, ErrorType, ErrorSeverity, RecoveryStrategy, ApplicationError
    
    if clear:
        with console.status("[bold blue]Clearing error history...", spinner="dots"):
            default_handler.clear_errors()
            console.print("[bold green]Error history cleared![/]")
    
    if test:
        # Run a test with different error scenarios to demonstrate the error handling system
        console.print("[bold blue]Running error handling tests...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[bold blue]Testing error handling...", total=3)
            
            # Test file system error
            progress.update(task, description="[bold blue]Testing file system error...")
            try:
                with default_handler.error_context(
                    "file_operation",
                    error_types=[FileNotFoundError],
                    error_type=ErrorType.FILE_SYSTEM,
                    severity=ErrorSeverity.WARNING
                ):
                    # Intentionally try to open a non-existent file
                    with open("non_existent_file.txt", "r") as f:
                        content = f.read()
                
                console.print("[bold red]Test failed: Expected error was not raised[/]")
            except Exception as e:
                console.print("[green]File system error test completed successfully[/]")
            
            progress.update(task, advance=1)
            
            # Test retry strategy
            progress.update(task, description="[bold blue]Testing retry strategy...")
            
            # Define a function that fails the first time but succeeds on second attempt
            attempt_count = [0]
            
            @default_handler.with_error_handling(
                error_types=[ValueError],
                recovery_strategy=RecoveryStrategy.RETRY,
                error_type=ErrorType.DATA,
                severity=ErrorSeverity.WARNING
            )
            def test_retry_function():
                attempt_count[0] += 1
                if attempt_count[0] == 1:
                    raise ValueError("First attempt always fails")
                return "Success on retry"
            
            result = test_retry_function()
            if result == "Success on retry" and attempt_count[0] > 1:
                console.print("[green]Retry strategy test completed successfully[/]")
            else:
                console.print("[bold red]Retry strategy test failed[/]")
            
            progress.update(task, advance=1)
            
            # Test custom application error
            progress.update(task, description="[bold blue]Testing custom error...")
            
            try:
                # Create and log a custom application error
                error = ApplicationError(
                    "This is a test error",
                    error_type=ErrorType.MODEL,
                    severity=ErrorSeverity.INFO,
                    recovery_strategy=RecoveryStrategy.NONE,
                    context={"test": True, "purpose": "demonstration"}
                )
                default_handler.handle_error(error)
                console.print("[green]Custom error test completed successfully[/]")
            except Exception as e:
                console.print(f"[bold red]Custom error test failed: {e}[/]")
                
            progress.update(task, advance=1)
    
    if report:
        # Generate and display error report
        report_data = default_handler.get_error_report()
        
        if report_data["total_errors"] > 0:
            # Display error summary
            summary_table = Table(title="Error Summary")
            summary_table.add_column("Category", style="cyan")
            summary_table.add_column("Count", style="yellow")
            
            summary_table.add_row("Total Errors", str(report_data["total_errors"]))
            
            for error_type, count in report_data["error_counts_by_type"].items():
                if count > 0:
                    summary_table.add_row(f"Type: {error_type}", str(count))
            
            for severity, count in report_data["error_counts_by_severity"].items():
                if count > 0:
                    summary_table.add_row(f"Severity: {severity}", str(count))
            
            console.print(summary_table)
            
            # Display detailed error list
            errors_table = Table(title="Recent Errors")
            errors_table.add_column("Time", style="blue")
            errors_table.add_column("Type", style="cyan")
            errors_table.add_column("Severity", style="yellow")
            errors_table.add_column("Message", style="red")
            errors_table.add_column("Recovery", style="green")
            
            # Show the 10 most recent errors
            for error in report_data["errors"][-10:]:
                # Format timestamp
                timestamp = error["timestamp"].split("T")[1].split(".")[0]  # Extract time
                
                errors_table.add_row(
                    timestamp,
                    error["error_type"],
                    error["severity"],
                    error["message"][:50] + ("..." if len(error["message"]) > 50 else ""),
                    error["recovery_strategy"]
                )
            
            console.print(errors_table)
        else:
            console.print("[green]No errors have been recorded.[/]")
    
    if save:
        with console.status(f"[bold blue]Saving error report to {save}...", spinner="dots"):
            default_handler.save_error_report(save)
            console.print(f"[bold green]Error report saved to {save}![/]")

@app.callback()
def main():
    """
    AnimatePodcasting CLI - Create animated videos from podcast audio.
    """
    # Display ASCII art banner
    banner = """
[bold cyan]    _          _                 _       ___          _                  _   _             
   / \\   _ __ (_)_ __ ___   __ _| |_ ___|  _ \\ ___  __| | ___ __ _ ___| |_(_)_ __   __ _ 
  / _ \\ | '_ \\| | '_ ` _ \\ / _` | __/ _ \\ |_) / _ \\/ _` |/ __/ _` / __| __| | '_ \\ / _` |
 / ___ \\| | | | | | | | | | | (_| | ||  __/  __/ (_) | (_| | (_| (_| \\__ \\ |_| | | | | (_| |
/_/   \\_\\_| |_|_|_| |_| |_|\\__,_|\\__\\___|_|   \\___/ \\__,_|\\___\\__,_|___/\\__|_|_| |_|\\__, |
                                                                                     |___/ [/]
    """
    console.print(banner)

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Add exception handler for uncaught exceptions
    try:
        app()
    except Exception as e:
        if "default_handler" in sys.modules.get("src.error_handler", {}).__dict__:
            # Try to use our error handler for uncaught exceptions
            from src.error_handler import default_handler, ErrorType, ErrorSeverity
            default_handler.handle_error(
                e, 
                context={"location": "main_app_entry"},
                recovery_strategy=RecoveryStrategy.NONE
            )
        
        # Display error in a nice format
        console.print("")
        console.print(Panel(
            f"[bold red]Error:[/] {str(e)}\n\n[dim]{traceback.format_exc()}[/dim]",
            title="Unhandled Exception",
            border_style="red"
        ))
        sys.exit(1) 