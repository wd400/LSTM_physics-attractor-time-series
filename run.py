from typing import Dict, List, Tuple, Any, Optional
import argparse
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
import yaml
from numpy.typing import NDArray

# for reproducibility especially splitting
np.random.seed(42)
tf.random.set_seed(42)


base_path = os.path.dirname(os.path.realpath(__file__))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ML pipeline runner')
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['split', 'train', 'eval'],
                      help='Mode to run: split, train, or eval')
    

    


    parser.add_argument('--dataset_path', type=str,
                      help='Path to dataset for split mode',
                        default= os.path.join(base_path, 'data')
                        )
    
    parser.add_argument('--split_path', type=str,
                        help='Path to save split data',
                        default= os.path.join(base_path, 'split'))
    parser.add_argument('--model_path', type=str,
                        help='Path to save model data',
                        default= os.path.join(base_path, 'model'))
    parser.add_argument('--eval_path', type=str,
                        help='Path to save eval data',
                        default= os.path.join(base_path, 'eval'))
    
    parser.add_argument('--hyperparams', type=str,
                      help='Path to hyperparameters config file')
    parser.add_argument('--mlflow_tracking_uri', type=str, 
                      help='MLflow tracking URI')
    parser.add_argument('--experiment_id', type=str,
                      help='MLflow experiment ID')
    
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def split_dataset(dataset_path: str, split_path: str) -> None:
    """Split dataset into train/val/test sets and save to disk"""
    # Create split directory if it doesn't exist
    os.makedirs(split_path, exist_ok=True)
    
    # Load dataset
    df: pd.DataFrame = pd.read_csv(f"{dataset_path}/data.csv")
    df = df.iloc[:, :-8]
    df = df.drop(df.columns[0], axis=1)
    
    # Split indices
    train_index: int = round(len(df.index) * 0.6)
    val_index: int = round(len(df.index) * 0.8)
    
    # Split data
    train_df: pd.DataFrame = df[:train_index]
    val_df: pd.DataFrame = df[train_index:val_index]
    test_df: pd.DataFrame = df[val_index:]
    
    # Save splits
    train_df.to_csv(os.path.join(split_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(split_path, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(split_path, 'test.csv'), index=False)

def create_sequences(data: NDArray[np.float64], n_past: int, n_future: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create sequences for time series prediction"""
    X: List[NDArray[np.float64]] = []
    y: List[NDArray[np.float64]] = []
    for window_start in range(len(data)):
        past_end: int = window_start + n_past
        future_end: int = past_end + n_future
        if future_end > len(data):
            break
        X.append(data[window_start:past_end, :])
        y.append(data[past_end:future_end, :])
    return np.array(X), np.array(y)

def build_model(n_past: int, n_future: int, n_features: int) -> Model:
    """Build the encoder-decoder model"""
    encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(10, return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    
    encoder_l2 = tf.keras.layers.LSTM(10, return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]
    
    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
    
    decoder_l1 = tf.keras.layers.LSTM(10, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(10, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
    
    return tf.keras.models.Model(encoder_inputs, decoder_outputs2)

def train_model(split_path: str, model_path: str, config: Dict[str, Any]) -> None:
    """Train the model and save it"""
    # Set up MLflow
    mlflow.tensorflow.autolog()

    mlflow.log_params(config)

    # copy config to model directory for tracking
    with open(os.path.join(model_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    
    # set seed for reproducibility
    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])
    
    
    # Load data
    train_df: pd.DataFrame = pd.read_csv(os.path.join(split_path, 'train.csv'))
    val_df: pd.DataFrame = pd.read_csv(os.path.join(split_path, 'val.csv'))
    
    # Scale data
    scalers: Dict[str, MinMaxScaler] = {}
    for column in train_df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df[column] = scaler.fit_transform(train_df[column].values.reshape(-1, 1))
        val_df[column] = scaler.transform(val_df[column].values.reshape(-1, 1))
        scalers[f'scaler_{column}'] = scaler
    
    # Save scalers
    np.save(os.path.join(model_path, 'scalers.npy'), scalers)
    
    # Create sequences
    n_past: int = config['n_past']
    n_future: int = config['n_future']
    n_features: int = train_df.shape[1]
    
    X_train, y_train = create_sequences(train_df.values, n_past, n_future)
    X_val, y_val = create_sequences(val_df.values, n_past, n_future)
    
    # Build and train model
    model: Model = build_model(n_past, n_future, n_features)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                 loss=tf.keras.losses.Huber())
    
    history = model.fit(
        X_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                lambda x: config['learning_rate'] * 0.90 ** x
            )
        ]
    )
    
    # Save model
    model.save(os.path.join(model_path, 'model.keras'))
    
    # Log metrics
    mlflow.log_metrics({
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1]
    })

def evaluate_model(split_path: str, model_path: str, eval_path: str) -> None:
    """Evaluate the model and save results"""
    # Load config
    config: Dict[str, Any] = load_config(os.path.join(model_path, 'config.yaml'))

    # Load test data
    test_df: pd.DataFrame = pd.read_csv(os.path.join(split_path, 'test.csv'))
    
    # Load scalers and scale test data
    scalers: Dict[str, MinMaxScaler] = np.load(os.path.join(model_path, 'scalers.npy'), allow_pickle=True).item()
    for column in test_df.columns:
        scaler = scalers[f'scaler_{column}']
        test_df[column] = scaler.transform(test_df[column].values.reshape(-1, 1))
    
    # Create sequences
    n_past: int = config['n_past']
    n_future: int = config['n_future']
    X_test, y_test = create_sequences(test_df.values, n_past, n_future)
    
    # Load model and evaluate
    model: Model = tf.keras.models.load_model(os.path.join(model_path, 'model.keras'))
    test_loss: float = model.evaluate(X_test, y_test)
    
    # Log metrics
    mlflow.log_metric('test_loss', test_loss)

def main() -> None:
    args: argparse.Namespace = parse_args()

    
    # Set up MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if args.experiment_id:
        mlflow.set_experiment(args.experiment_id)
    else:
        mlflow.set_experiment('default')
    
    # Load config if provided
    config: Optional[Dict[str, Any]] = None
    if args.hyperparams:
        config = load_config(args.hyperparams)
    
    # Execute requested mode
    if args.mode == 'split':
        if not args.dataset_path:
            raise ValueError("dataset_path is required for split mode")
        split_dataset(args.dataset_path, args.split_path)
    
    elif args.mode == 'train':
        if not config:
            raise ValueError("hyperparams config is required for train mode")
        os.makedirs( args.model_path, exist_ok=True)
            
        train_model( args.split_path
            , 
            args.model_path
                    
                    ,
                      config)
    
    elif args.mode == 'eval':
        os.makedirs(
            args.eval_path, exist_ok=True)
        evaluate_model(
            args.split_path, args.model_path, args.eval_path)

if __name__ == "__main__":
    main()