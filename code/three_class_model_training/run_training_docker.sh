#!/bin/bash

# =============================================================================
# Docker Three-Class Training Runner Script
# =============================================================================
# Runs three-class classification training (cassette vs alt_three vs alt_five)
# in Docker container (detached mode) 
# Container runs in background 
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${SCRIPT_DIR}/three_class_model_training"
DOCKER_IMAGE="aspect-gpu"
CONTAINER_NAME="aspect-three-class-training-$(date +%Y%m%d-%H%M%S)"
PROJECT_DIR="/app"
RESULTS_DIR="${PROJECT_DIR}/three_class_model_training/result_sample_2"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


# Function to display usage
usage() {
    cat << EOF
Usage: $0 [ACTION] [OPTIONS]

Run three-class classification training (cassette vs alt_three vs alt_five) in Docker container.

ACTIONS:
    start [OPTIONS]     Start training in detached Docker container (background)
    logs [CONTAINER]    View live logs from container (default: latest)
    stop [CONTAINER]    Stop a running container
    status              Show status of all training containers
    exec [CONTAINER]    Execute bash in container
    list                List all training containers
    clean               Stop and remove all training containers

OPTIONS (for start action):
    -g, --gpu GPU_ID            GPU ID (default: 0)
    -t, --trials N              Number of Optuna trials (default: 20)
    -b, --batch-size N          Batch size (default: 32)
    -e, --epochs N              Training epochs (default: 30)
    -lr, --learning-rate LR     Learning rate (default: 1e-4)
    --name NAME                 Custom container name

EXAMPLES:
    # Start training with default settings (GPU 0, 20 Optuna trials)
    $0 start

    # Start training on GPU 1 with 30 Optuna trials
    $0 start -g 1 -t 30

    # View live logs
    $0 logs

    # View logs from specific container
    $0 logs aspect-three-class-training-20250105-120000

    # Stop a container
    $0 stop aspect-three-class-training-20250105-120000

    # Check status
    $0 status

    # Execute bash in container
    $0 exec

EOF
}

# Check if Docker image exists
check_docker_image() {
    if ! docker images | grep -q "^${DOCKER_IMAGE}"; then
        print_warning "Docker image '${DOCKER_IMAGE}' not found. Building..."
        docker build -t ${DOCKER_IMAGE} -f ${SCRIPT_DIR}/dockerfile ${SCRIPT_DIR}
        print_success "Docker image built successfully"
    fi
}

# Start training in Docker
start_training() {
    local GPU_ID=0
    local OPTUNA_TRIALS=20
    local BATCH_SIZE=32
    local EPOCHS=30
    local LEARNING_RATE="1e-4"
    local CUSTOM_NAME=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -g|--gpu)
                GPU_ID="$2"
                shift 2
                ;;
            -t|--trials)
                OPTUNA_TRIALS="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -e|--epochs)
                EPOCHS="$2"
                shift 2
                ;;
            -lr|--learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --name)
                CUSTOM_NAME="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Use custom name if provided
    if [[ -n "$CUSTOM_NAME" ]]; then
        CONTAINER_NAME="$CUSTOM_NAME"
    fi

    # Check if container name already exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        print_error "Container '${CONTAINER_NAME}' already exists. Please use a different name or remove it first."
        exit 1
    fi

    check_docker_image

    # Dataset path (relative to project directory)
    DATASET_PATH="${PROJECT_DIR}/data_preprocessing/balanced_three_class_from_multiclass"

    # Build training command
    TRAIN_CMD="cd '${PROJECT_DIR}/three_class_model_training' && python3 three_class_training.py \
        --model_name_or_path 'zhihan1996/DNABERT-2-117M' \
        --data_path '${DATASET_PATH}' \
        --output_dir '${RESULTS_DIR}' \
        --logging_dir '${RESULTS_DIR}/logs' \
        --use_class_weights \
        --use_optuna \
        --optuna_trials ${OPTUNA_TRIALS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --num_train_epochs ${EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay 0.01 \
        --warmup_steps 500 \
        --logging_steps 50 \
        --save_steps 100 \
        --eval_steps 100 \
        --evaluation_strategy 'epoch' \
        --save_strategy 'epoch' \
        --load_best_model_at_end \
        --metric_for_best_model 'eval_combined_score' \
        --greater_is_better \
        --save_total_limit 1 \
        --fp16 \
        --seed 42 \
        --report_to ''"

    print_info "=============================================================================="
    print_info "Starting Three-Class Model Training in Docker (Detached Mode)"
    print_info "=============================================================================="
    print_info "Container Name:     ${CONTAINER_NAME}"
    print_info "GPU ID:             ${GPU_ID}"
    print_info "Dataset:            cassette_alt_three_alt_five (balanced)"
    print_info "Optuna Trials:      ${OPTUNA_TRIALS}"
    print_info "Batch Size:         ${BATCH_SIZE}"
    print_info "Epochs:             ${EPOCHS}"
    print_info "Learning Rate:     ${LEARNING_RATE}"
    print_info "Focus:              Extra attention on alt_five class"
    print_info "Results Dir:        ${RESULTS_DIR}"
    print_info "=============================================================================="
    echo ""

    GPU_OPTION="device=${GPU_ID}"
    print_info "Starting container in detached mode on GPU ${GPU_ID}..."
    
    docker run -d \
        --gpus "${GPU_OPTION}" \
        --restart=no \
        --name ${CONTAINER_NAME} \
        -v "${SCRIPT_DIR}:${PROJECT_DIR}" \
        -e NVIDIA_VISIBLE_DEVICES=${GPU_ID} \
        -e TOKENIZERS_PARALLELISM=false \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e RESULTS_DIR=${RESULTS_DIR} \
        ${DOCKER_IMAGE} \
        bash -c "${TRAIN_CMD}"

    if [[ $? -eq 0 ]]; then
        print_success "Container started successfully in detached mode!"
        echo ""
        print_info "Container Name: ${CONTAINER_NAME}"
        print_info "Container will continue running even if you disconnect!"
        echo ""
        print_info "To view logs:"
        print_info "  $0 logs ${CONTAINER_NAME}"
        print_info "  Or: docker logs -f ${CONTAINER_NAME}"
        echo ""
        print_info "To check status:"
        print_info "  $0 status"
        print_info "  Or: docker ps | grep ${CONTAINER_NAME}"
        echo ""
        print_info "To stop training:"
        print_info "  $0 stop ${CONTAINER_NAME}"
        print_info "  Or: docker stop ${CONTAINER_NAME}"
        echo ""
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# View logs
view_logs() {
    local CONTAINER="$1"
    
    if [[ -z "$CONTAINER" ]]; then
        # Find the most recent container
        CONTAINER=$(docker ps -a --filter "name=aspect-three-class-training-" --format "{{.Names}}" | sort -r | head -1)
        if [[ -z "$CONTAINER" ]]; then
            print_error "No training containers found"
            exit 1
        fi
        print_info "Using latest container: $CONTAINER"
    fi

    if ! docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
        print_error "Container '$CONTAINER' not found"
        exit 1
    fi

    print_info "Viewing logs from container: $CONTAINER"
    print_info "Press Ctrl+C to stop viewing logs (container will continue running)"
    echo ""
    docker logs -f ${CONTAINER}
}

# Stop container
stop_container() {
    local CONTAINER="$1"
    
    if [[ -z "$CONTAINER" ]]; then
        CONTAINER=$(docker ps --filter "name=aspect-three-class-training-" --format "{{.Names}}" | sort -r | head -1)
        if [[ -z "$CONTAINER" ]]; then
            print_error "No running training containers found"
            exit 1
        fi
        print_info "Using latest running container: $CONTAINER"
    fi

    if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
        print_error "Container '$CONTAINER' is not running"
        exit 1
    fi

    print_info "Stopping container: $CONTAINER"
    docker stop ${CONTAINER}
    print_success "Container stopped"
}

# Show status
show_status() {
    print_info "Training containers status:"
    echo ""
    docker ps -a --filter "name=aspect-three-class-training-" --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | head -1
    docker ps -a --filter "name=aspect-three-class-training-" --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | tail -n +2 | sort -r
}

# Execute bash in container
exec_container() {
    local CONTAINER="$1"
    
    if [[ -z "$CONTAINER" ]]; then
        CONTAINER=$(docker ps --filter "name=aspect-three-class-training-" --format "{{.Names}}" | sort -r | head -1)
        if [[ -z "$CONTAINER" ]]; then
            print_error "No running training containers found"
            exit 1
        fi
        print_info "Using latest running container: $CONTAINER"
    fi

    if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
        print_error "Container '$CONTAINER' is not running"
        exit 1
    fi

    print_info "Executing bash in container: $CONTAINER"
    docker exec -it ${CONTAINER} bash
}

# List containers
list_containers() {
    print_info "All training containers:"
    echo ""
    docker ps -a --filter "name=aspect-three-class-training-" --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}"
}

# Clean all containers
clean_containers() {
    print_warning "This will stop and remove ALL training containers!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cancelled"
        return
    fi

    local CONTAINERS=$(docker ps -a --filter "name=aspect-three-class-training-" --format "{{.Names}}")
    if [[ -z "$CONTAINERS" ]]; then
        print_info "No containers to clean"
        return
    fi

    print_info "Stopping and removing containers..."
    echo "$CONTAINERS" | xargs docker stop 2>/dev/null || true
    echo "$CONTAINERS" | xargs docker rm 2>/dev/null || true
    print_success "All containers cleaned"
}

# Main script logic
case "${1:-}" in
    start)
        shift
        start_training "$@"
        ;;
    logs)
        shift
        view_logs "$@"
        ;;
    stop)
        shift
        stop_container "$@"
        ;;
    status)
        show_status
        ;;
    exec)
        shift
        exec_container "$@"
        ;;
    list)
        list_containers
        ;;
    clean)
        clean_containers
        ;;
    *)
        usage
        exit 1
        ;;
esac

