# Docker Usage Guide

AnimatePodcasting can be easily containerized and run in Docker, providing a consistent environment across different systems.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)

## Building the Docker Image

You can build the Docker image using one of the following methods:

### Using the build script

```bash
./docker_build.sh
```

### Using Docker directly

```bash
docker build -t animate_podcasting:latest .
```

### Using Docker Compose

```bash
docker-compose build
```

## Running the Container

### Using Docker Compose (Recommended)

The easiest way to run the container is using Docker Compose:

```bash
docker-compose up
```

This will start the container with the default command defined in `docker-compose.yml`. To run a different command, modify the `command` section in the compose file.

### Using Docker directly

You can also run the container directly using Docker:

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  animate_podcasting:latest [command]
```

Replace `[command]` with the actual command you want to run.

## Example Commands

### Show help

```bash
docker run -it --rm animate_podcasting:latest --help
```

### Transcribe an audio file

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  animate_podcasting:latest transcribe data/your_audio.mp3
```

### Run full pipeline

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  animate_podcasting:latest create data/your_audio.mp3 --model base --num-images 5
```

### Run tests

```bash
docker run -it --rm animate_podcasting:latest ./run_tests.py
```

## Volume Mounts

The container uses the following volume mounts:

- `/app/data`: Input data directory (audio files)
- `/app/outputs`: Output directory (transcriptions, images, videos)
- `/app/logs`: Log files
- `/app/cache`: Cache directory for improved performance

## Environment Variables

You can pass environment variables to the container:

```bash
docker run -it --rm \
  -e NOTION_TOKEN=your_token \
  -e NOTION_DATABASE_ID=your_database_id \
  animate_podcasting:latest [command]
```

Or in docker-compose.yml:

```yaml
services:
  animate:
    # ... other settings ...
    environment:
      - NOTION_TOKEN=your_token
      - NOTION_DATABASE_ID=your_database_id
```

## Troubleshooting

### Permission issues

If you encounter permission issues with the mounted volumes, ensure that the directories have the correct permissions:

```bash
chmod -R 755 data outputs logs cache
```

### Container crashes

Check the logs for more information:

```bash
docker logs animate_podcasting
```

### Performance issues

Ensure you've allocated enough resources to Docker, especially when using the larger Whisper models or Stable Diffusion.

## Advanced Usage

### Custom Entrypoint

You can override the entrypoint to use a different script:

```bash
docker run -it --rm --entrypoint ./run_tests.py animate_podcasting:latest
```

### Running in Background

To run the container in the background:

```bash
docker-compose up -d
```

Or with Docker:

```bash
docker run -d \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  animate_podcasting:latest [command]
``` 