#!/bin/bash
# Build the Docker image for AnimatePodcasting

echo "Building AnimatePodcasting Docker image..."
docker build -t animate_podcasting:latest .

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can now run the container with:"
    echo "  docker run -it --rm -v \$(pwd)/data:/app/data -v \$(pwd)/outputs:/app/outputs animate_podcasting:latest [command]"
    echo ""
    echo "Example commands:"
    echo "  docker run -it --rm animate_podcasting:latest --help"
    echo "  docker run -it --rm -v \$(pwd)/data:/app/data -v \$(pwd)/outputs:/app/outputs animate_podcasting:latest transcribe data/your_audio.mp3"
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose up"
    echo ""
else
    echo "Build failed!"
fi 