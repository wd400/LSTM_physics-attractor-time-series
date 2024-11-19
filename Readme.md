# Time Series Prediction Pipeline

This project is a time series prediction pipeline using TensorFlow and MLflow. The pipeline includes data splitting, model training, and evaluation.

## Setup

### Prerequisites

- Python 3.12
- Docker (optional, for containerized execution)

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configuration

Edit the `config.yaml` file to set the model architecture and training parameters.

### Running the Pipeline

The pipeline can be run in three modes: `split`, `train`, and `eval`.

1. **Split the dataset**:
    ```sh
    python run.py --mode split --dataset_path <path-to-dataset> --split_path <path-to-save-splits>
    ```

2. **Train the model**:
    ```sh
    python run.py --mode train --split_path <path-to-splits> --model_path <path-to-save-model> --hyperparams config.yaml
    ```

3. **Evaluate the model**:
    ```sh
    python run.py --mode eval --split_path <path-to-splits> --model_path <path-to-model> --eval_path <path-to-save-evaluation>
    ```

### Using Docker

Build the Docker image:
```sh
docker build -t time-series-pipeline .
```