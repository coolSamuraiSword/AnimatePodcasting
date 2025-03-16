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
