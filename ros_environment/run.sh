#!/bin/bash

# Function to display usage
show_usage() {
    echo "Usage: $0 [build|up|down|logs]"
    echo "  build  - Build the Docker image"
    echo "  up     - Start the container and show logs"
    echo "  down   - Stop the container"
    echo "  logs   - Show container logs"
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Process commands
case "$1" in
    build)
        echo "Building Docker image..."
        docker-compose build
        ;;
    up)
        echo "Starting container..."
        docker-compose up
        ;;
    down)
        echo "Stopping container..."
        docker-compose down
        ;;
    logs)
        echo "Showing container logs..."
        docker-compose logs -f
        ;;
    *)
        show_usage
        exit 1
        ;;
esac 