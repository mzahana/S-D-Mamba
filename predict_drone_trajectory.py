"""
(c) Mohamed Abdelkader 2024

Usage

    python predict_drone_trajectory.py --model_path checkpoints/drone_traj_exp_20_10_S_Mamba_DroneTraj/checkpoint.pth --scaler_path dataset/drone_traj/drone_scaler.pkl --csv_file dataset/drone_traj/gazebo_trajectory_1.csv --seq_len 20 --pred_len 10 --model S_Mamba --num_runs 10
"""
import argparse
import torch
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from model import S_Mamba  # Import the S_Mamba model class
import plotly.graph_objects as go
import os


def load_model(model_path, args, device):
    """
    Load the model from the specified path.
    """
    model_class = getattr(S_Mamba, 'Model')  # Dynamically get the model class
    model = model_class(args).to(device)  # Create the model object
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model state
    model.eval()  # Set model to evaluation mode
    return model


def prepare_data(csv_file, seq_len, pred_len, scaler):
    """
    Load the CSV file and prepare the data for model input.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Extract the relevant columns for position and velocity
    data = data[['tx', 'ty', 'tz', 'vx', 'vy', 'vz']]

    # Normalize the data
    data_scaled = scaler.transform(data)

    # Randomly select a starting point for the input sequence
    max_start = len(data) - seq_len - pred_len
    start_idx = np.random.randint(0, max_start)
    end_idx = start_idx + seq_len
    label_end_idx = end_idx + pred_len

    # Prepare input and true output sequences
    input_seq = data_scaled[start_idx:end_idx]
    true_seq = data_scaled[end_idx:label_end_idx]

    return input_seq, true_seq, data.iloc[start_idx:end_idx], data.iloc[end_idx:label_end_idx]


def plot_trajectory(true_data, input_data, predicted_data, rmse_value, output_path="interactive_trajectory_plot.html"):
    """
    Plot the 3D position trajectory, comparing true, input, and predicted data.
    Save the interactive plot to a file.
    """
    fig = go.Figure()

    # Plot true trajectory in blue
    fig.add_trace(go.Scatter3d(
        x=true_data['tx'], y=true_data['ty'], z=true_data['tz'],
        mode='lines', name='True Trajectory', line=dict(color='blue', width=4)
    ))

    # Plot input sequence in green
    fig.add_trace(go.Scatter3d(
        x=input_data['tx'], y=input_data['ty'], z=input_data['tz'],
        mode='lines', name='Input Sequence', line=dict(color='green', width=4)
    ))

    # Plot predicted trajectory in red
    fig.add_trace(go.Scatter3d(
        x=predicted_data[:, 0], y=predicted_data[:, 1], z=predicted_data[:, 2],
        mode='lines', name='Predicted Trajectory', line=dict(color='red', width=4)
    ))

    # Set the title and axis labels
    fig.update_layout(
        title=f"3D Drone Trajectory with RMSE: {rmse_value:.4f}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Save the interactive plot as an HTML file
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")

    # Display the figure
    fig.show()


def main(model_path, scaler_path, csv_file, seq_len, pred_len, args, num_runs=10):
    # Check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the scaler for normalization
    scaler = joblib.load(scaler_path)

    # Load the trained model
    model = load_model(model_path, args, device)

    # Prepare the data from the CSV file
    input_seq, true_seq, input_data, true_data = prepare_data(csv_file, seq_len, pred_len, scaler)

    # Convert input sequence to a tensor and move it to the appropriate device
    input_seq_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

    # Prepare a placeholder for the decoder input (zeros for the prediction length)
    dec_inp = torch.zeros((1, pred_len, input_seq.shape[1]), dtype=torch.float32).to(device)

    # Perform inference multiple times and measure average inference time
    inference_times = []
    for run in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            predicted_seq = model(input_seq_tensor, None, dec_inp, None)  # Forward pass through the model
        end_time = time.time()

        inference_time = end_time - start_time
        inference_times.append(inference_time)

    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time over {num_runs} runs: {avg_inference_time:.4f} seconds")

    # Move prediction back to CPU and denormalize
    predicted_seq = predicted_seq.squeeze(0).cpu().numpy()
    predicted_seq_denorm = scaler.inverse_transform(predicted_seq)

    # Calculate RMSE between true and predicted trajectories (3D positions)
    rmse_value = np.sqrt(mean_squared_error(true_data[['tx', 'ty', 'tz']].values, predicted_seq_denorm[:, :3]))
    print(f"RMSE: {rmse_value:.4f}")

    # Plot the results (interactive 3D plot)
    plot_trajectory(true_data, input_data, predicted_seq_denorm, rmse_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model .pth file')
    parser.add_argument('--scaler_path', type=str, required=True, help='Path to the scaler .pkl file')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing drone trajectory')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length for the input')
    parser.add_argument('--pred_len', type=int, required=True, help='Prediction length')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., S_Mamba)')

    # Optional model parameters (with default values)
    parser.add_argument('--is_training', type=int, default=1, help='Training status (not used in inference)')
    parser.add_argument('--enc_in', type=int, default=6, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=6, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=6, help='Output size')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=512, help='Dimension of feedforward network')
    parser.add_argument('--d_state', type=int, default=2, help='State size of Mamba block')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--output_attention', type=bool, default=False, help='Whether to output attention')
    parser.add_argument('--use_norm', type=int, default=1, help='Use normalization and de-normalization')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--class_strategy', type=str, default='projection', help='Class strategy')
    parser.add_argument('--embed', type=str, default='timeF', help='Embedding type (timeF, fixed, learned)')
    parser.add_argument('--freq', type=str, default='h', help='Frequency for time features encoding')

    # Optional argument for number of inference runs
    parser.add_argument('--num_runs', type=int, default=10, help='Number of times to run inference for averaging time')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments and num_runs for averaging
    main(args.model_path, args.scaler_path, args.csv_file, args.seq_len, args.pred_len, args, num_runs=args.num_runs)
