# AnimatePodcasting

Generate animated videos from podcast audio using AI-powered transcription, content analysis, and image generation.

## Features

- **Audio Transcription**: Convert podcast audio to text using OpenAI's Whisper model
- **Content Analysis**: Extract key segments from transcriptions using NLP
- **Image Generation**: Create images based on key segments using Stable Diffusion
- **Video Creation**: Combine images with audio to create animated videos
- **Caching System**: Improve performance with smart caching of expensive operations
- **Beautiful CLI**: User-friendly command-line interface with rich formatting
- **Notion Integration**: Track project progress in Notion workspace
- **Performance Monitoring**: Track and benchmark performance of critical operations
- **Error Handling & Recovery**: Robust error handling with automatic recovery strategies
- **Testing Framework**: Comprehensive test suite with rich reporting
- **Docker Support**: Easy containerization for consistent deployment

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd AnimatePodcasting
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Configure Notion integration:
   - Create a `.env` file with your Notion credentials:
     ```
     NOTION_TOKEN=your_notion_token
     NOTION_DATABASE_ID=your_database_id
     ```

## Usage

The project provides a beautiful command-line interface:

```bash
# Full pipeline: transcribe audio, analyze content, generate images, create video
./animate.py create your_audio.mp3 --model base --num-images 5

# Individual steps
./animate.py transcribe your_audio.mp3 --output transcript.json
./animate.py analyze transcript.json --num-segments 5
./animate.py generate "A beautiful landscape" --output image.png

# Cache management
./animate.py cache
./animate.py cache --clear

# Performance monitoring
./animate.py performance --benchmark
./animate.py performance --report --save reports/performance.json

# Error handling
./animate.py errors --report
./animate.py errors --test
```

## Testing

The project includes a comprehensive test suite to ensure reliability:

```bash
# Run all tests with beautiful reporting
./run_tests.py

# Run specific test file
python -m unittest tests/test_performance.py

# Run with coverage report
coverage run -m unittest discover tests
coverage report
```

## Docker

The project can be easily containerized using Docker:

```bash
# Build Docker image
./docker_build.sh

# Run with Docker Compose
docker-compose up

# Run specific command in container
docker run -it --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs animate_podcasting:latest transcribe data/your_audio.mp3
```

See [Docker Usage Guide](docs/docker.md) for more details.

## Development

This project uses Git for version control. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone the repository and set up the project
git clone <repository-url>
cd AnimatePodcasting

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and test
./animate.py version

# Commit your changes with a meaningful message
git add -A
git commit -m "Add feature: description of your changes"

# Push changes to remote
git push origin feature/your-feature-name
```

## Project Structure

- `animate.py`: Main CLI application
- `src/`: Core modules
  - `transcriber.py`: Audio transcription with Whisper
  - `analyzer.py`: Content analysis and key segment extraction
  - `generator.py`: Image generation with Stable Diffusion
  - `cache_manager.py`: Caching system for performance optimization
  - `project_notion.py`: Notion integration for project tracking
- `data/`: Directory for input audio files
- `outputs/`: Directory for generated outputs (transcripts, images, videos)

## License

MIT
