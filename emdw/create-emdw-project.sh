#!/bin/bash

# A script to initialize a new "tut" C++ project in a specified directory.

# --- Configuration ---
# Source locations for the project template.
# Modify these paths if your template files are located elsewhere.
TEMPLATE_ARCHIVE="$HOME/devel/emdw/extras/my_emdw_app.tgz"
EXAMPLE_SRC="$HOME/devel/emdw/src/bin/example.cc"

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Script Logic ---

# Step 1: Validate input
echo -e "${YELLOW}=== Starting New Project Setup ===${NC}"
if [ $# -ne 1 ]; then
    echo -e "${RED}Error: Please provide a path for the new project directory.${NC}"
    echo -e "Usage: $0 ./path/to/my-new-project"
    exit 1
fi

PROJECT_DIR=$1

if [ -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Error: Directory '$PROJECT_DIR' already exists. Please choose a different name.${NC}"
    exit 1
fi

echo -e "Project will be created in: ${BLUE}$PROJECT_DIR${NC}"

# Step 2: Create directory and extract the project template
echo -e "\n${BLUE}1. Creating directory and extracting template...${NC}"
mkdir -p "$PROJECT_DIR" || { echo -e "${RED}Failed to create directory!${NC}"; exit 1; }

# The --strip-components=1 flag removes the top-level 'my_emdw_app/' folder from the archive,
# placing its contents directly into our new project directory.
tar -xvzf "$TEMPLATE_ARCHIVE" --strip-components=1 -C "$PROJECT_DIR" || { echo -e "${RED}Extraction failed!${NC}"; exit 1; }

# Step 3: Replace the default source file with the example
TARGET_SRC="$PROJECT_DIR/src/my_emdw_app.cc"
echo -e "\n${BLUE}2. Replacing template source file with example...${NC}"

if [ ! -f "$TARGET_SRC" ]; then
    echo -e "${RED}Error: Expected source file not found at '$TARGET_SRC'. Aborting.${NC}"
    exit 1
fi

cp "$EXAMPLE_SRC" "$TARGET_SRC" || { echo -e "${RED}Copying example source failed!${NC}"; exit 1; }
echo -e "Successfully replaced 'src/my_emdw_app.cc'."

# Step 4: Navigate into the new project and build with CMake
echo -e "\n${BLUE}3. Changing to project directory and running CMake...${NC}"
cd "$PROJECT_DIR" || { echo -e "${RED}Failed to navigate into '$PROJECT_DIR'!${NC}"; exit 1; }

cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON || { echo -e "${RED}CMake build configuration failed!${NC}"; exit 1; }

# Step 5: Add the custom 'run' target to the generated Makefile
MAKEFILE_PATH="./build/Makefile"
echo -e "\n${BLUE}4. Adding custom 'run' target to Makefile...${NC}"
if [ -f "$MAKEFILE_PATH" ]; then
    # Use printf to properly handle newline and tab characters
    printf '\nrun:\n\tsrc/my_emdw_app\n' >> "$MAKEFILE_PATH"
    echo -e "Successfully added 'run' target."
else
    echo -e "${RED}Warning: Makefile not found at '$MAKEFILE_PATH'. Could not add 'run' target.${NC}"
fi

echo -e "\n${GREEN}✔ Setup complete for project in '$PROJECT_DIR'!${NC}"
echo -e "You can now work inside the directory and run your build commands."

