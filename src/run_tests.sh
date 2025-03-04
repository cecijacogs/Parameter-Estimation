#!/bin/bash

# Navigate to the src directory (if needed)
cd "$(dirname "$0")/../src" || exit

# Run the tests using pytest
echo "Running tests..."
pytest

# Check the exit status of pytest to determine if tests passed
if [ $? -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi