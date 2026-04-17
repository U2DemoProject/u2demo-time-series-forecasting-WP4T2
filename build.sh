#!/bin/bash

# Build script for Docker image
# Builds a Docker image for the U2Demo time-series forecasting project.

set -e

# Function to display usage/help
show_help() {
    cat << EOF
Build script for Docker image

Usage: ./build.sh <tag> [--latest]

Arguments:
    tag         Tag for the Docker image (required)
                The image will be built as: u2demo/time-series-forecasting:<tag>

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

IMAGE_NAME="u2demo/time-series-forecasting:$TAG"

# requirements.txt is kept up-to-date by the poetry-export pre-commit hook,
# so we just verify it exists rather than regenerating it.
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found. Run 'poetry export -f requirements.txt --output requirements.txt --without-hashes' first."
    exit 1
fi

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build --platform linux/amd64 -t "$IMAGE_NAME" .

# Tag as latest if requested
if [ "$TAG_LATEST" = true ]; then
    echo "Tagging image as latest..."
    docker tag "$IMAGE_NAME" "u2demo/time-series-forecasting:latest"
fi

echo "Build complete! Image: $IMAGE_NAME"
if [ "$TAG_LATEST" = true ]; then
    echo "Also tagged as: u2demo/time-series-forecasting:latest"
fi

# Best-effort sanity check: verify the freshly-built image can launch the CLI.
# Runs without --gpus, so it only validates that dependencies load correctly.
# Failures here do not abort the build — the image is still built and tagged.
echo ""
echo "Running post-build sanity check..."
if docker run --rm "$IMAGE_NAME" sanity_check; then
    echo "Sanity check passed."
else
    echo "Warning: sanity check failed. The image was still built, but verify dependencies and CUDA availability before publishing."
fi
