#!/bin/bash

# Script to run finetuning analysis with different classifier types
# Usage: ./run_finetuning.sh [results_dir] [dataset] [model_type] [classifier_type]

# Default values
RESULTS_DIR=${1:-"results/latest"}  # Use "latest" to automatically find the most recent results
DATASET=${2:-"Actor"}                # Default dataset
MODEL_TYPE=${3:-"gcn"}              # Default model type
CLASSIFIER_TYPE=${4:-"logistic"}       # Default classifier type: 'ridge', 'logistic', or 'mlp'

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}====== Running Finetuning Analysis ======${NC}"
echo -e "${YELLOW}Dataset:${NC} $DATASET"
echo -e "${YELLOW}Model Type:${NC} $MODEL_TYPE"
echo -e "${YELLOW}Classifier Type:${NC} $CLASSIFIER_TYPE"

# Find the most recent results directory if "latest" is specified
if [ "$RESULTS_DIR" = "results/latest" ]; then
    echo -e "${YELLOW}Looking for the most recent results directory...${NC}"
    
    # Get the latest results directory matching the model type and dataset
    LATEST_DIR=$(find "results" -type d -name "${MODEL_TYPE}_${DATASET}_*" | sort -r | head -n 1)
    
    if [ -z "$LATEST_DIR" ]; then
        echo -e "${RED}Error: No matching results directory found for ${MODEL_TYPE}_${DATASET}_*${NC}"
        echo -e "${RED}Please specify a valid results directory or run main.py first.${NC}"
        exit 1
    fi
    
    RESULTS_DIR=$LATEST_DIR
    echo -e "${GREEN}Found results directory:${NC} $RESULTS_DIR"
fi

# Check if the specified results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory '$RESULTS_DIR' does not exist${NC}"
    exit 1
fi

# Check for model files
if [ ! -f "$RESULTS_DIR/f_model.pt" ] || [ ! -f "$RESULTS_DIR/g_model.pt" ]; then
    echo -e "${RED}Error: Model files not found in '$RESULTS_DIR'${NC}"
    echo -e "${RED}Make sure 'f_model.pt' and 'g_model.pt' exist in the directory.${NC}"
    exit 1
fi

# Prepare classifier-specific parameters
CLASSIFIER_PARAMS=""
if [ "$CLASSIFIER_TYPE" = "ridge" ]; then
    CLASSIFIER_PARAMS="--alpha 1e-7"  # Regularization strength
elif [ "$CLASSIFIER_TYPE" = "mlp" ]; then
    CLASSIFIER_PARAMS="--hidden_layers 128"  # MLP architecture
fi

# Run the finetuning script
echo -e "${BLUE}Running finetuning.py...${NC}"
echo "python finetuning.py --dataset $DATASET --model_type $MODEL_TYPE --results_dir $RESULTS_DIR --classifier_type $CLASSIFIER_TYPE $CLASSIFIER_PARAMS"

python finetuning.py \
    --dataset $DATASET \
    --model_type $MODEL_TYPE \
    --results_dir "$RESULTS_DIR" \
    --classifier_type $CLASSIFIER_TYPE \
    $CLASSIFIER_PARAMS

# Check if the command completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Finetuning analysis completed successfully!${NC}"
    echo -e "${GREEN}Results saved to:${NC} $RESULTS_DIR"
else
    echo -e "${RED}Error: Finetuning analysis failed.${NC}"
fi