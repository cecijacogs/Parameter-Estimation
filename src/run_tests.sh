#!/bin/bash

# Navigate to the directory of the script to ensure we're in the right folder
cd "$(dirname "$0")" || exit

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed. Please install dependencies first."
    exit 1
fi

# Run the tests (point to the tests directory)
echo "Running tests..."
pytest ../tests --maxfail=1 --disable-warnings -q

# Check if pytest succeeded or failed and exit accordingly
if [ $? -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi