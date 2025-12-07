#!/bin/bash
# Script to install and integrate ns3-gym opengym module

set -e

echo "Installing ns3-gym opengym module..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NS3_DIR="$SCRIPT_DIR"

# Check if ns3-gym directory exists
if [ ! -d "$NS3_DIR/ns3-gym" ]; then
    echo "Cloning ns3-gym repository..."
    cd "$NS3_DIR"
    git clone https://github.com/tkn-tub/ns3-gym.git
fi

# Copy opengym module to src directory
if [ -d "$NS3_DIR/ns3-gym/opengym" ]; then
    echo "Copying opengym module to src/..."
    cp -r "$NS3_DIR/ns3-gym/opengym" "$NS3_DIR/src/"
    echo "opengym module copied successfully!"
else
    echo "Error: opengym module not found in ns3-gym directory"
    echo "Please check if ns3-gym was cloned correctly"
    exit 1
fi

# Build opengym module
echo "Building opengym module..."
cd "$NS3_DIR"
./waf configure
./waf build

echo "opengym module installation complete!"
echo ""
echo "Next steps:"
echo "1. Make sure 'opengym' is added to dependencies in src/point-to-point/wscript"
echo "2. Rebuild the project: ./waf build"

