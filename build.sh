#!/bin/bash

# Build script for Docker image
# Builds a Docker image for the U2 time-series prediction project using poetry to export dependencies.

set -e

# Function to display usage/help
show_help() {
    cat << EOF
Build script for Docker image

Usage: ./build.sh <tag> [--latest]

Arguments:
    tag         Tag for the Docker image (required)
                The image will be built as: u2/time-series-prediction:<tag>

Options:
    --latest    Also tag the built image as 'latest'
    -h, --help  Show this help message

Examples:
    ./build.sh 1.0.0
    ./build.sh 1.0.0 --latest

EOF
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Check if tag argument is provided
if [ -z "$1" ]; then
    echo "Error: Tag parameter is required"
    echo ""
    show_help
    exit 1
fi

TAG=$1
TAG_LATEST=false

# Parse optional arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --latest)
            TAG_LATEST=true
            shift
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

IMAGE_NAME="u2/time-series-prediction:$TAG"

# Export requirements.txt using poetry
echo "Exporting requirements.txt using poetry..."
poetry export -f requirements.txt --output requirements.txt --without-hashes

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# Tag as latest if requested
if [ "$TAG_LATEST" = true ]; then
    echo "Tagging image as latest..."
    docker tag "$IMAGE_NAME" "u2/time-series-prediction:latest"
fi

echo "Build complete! Image: $IMAGE_NAME"
if [ "$TAG_LATEST" = true ]; then
    echo "Also tagged as: u2/time-series-prediction:latest"
fi

