import numpy as np
import pandas as pd
import torch
import sys
import torch.nn as nn
from mindxlib.explainers.timeseries import FDTempExplainer

# Test code
class LSTM_new_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, task='classification', ReVIn=False):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of LSTM layers.
            task (str): Task type, either 'classification' or 'regression'.
            ReVIn (bool): Whether to use reversible instance normalization.
        """
        super(LSTM_new_1, self).__init__()
        self.hidden_dim = hidden_dim
        self.task = task
        self.ReVIn = ReVIn
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.4)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states for LSTM.
        
        Returns:
            Tuple of hidden states (h0, c0).
        """
        return (torch.zeros(1, 256, self.hidden_dim),
                torch.zeros(1, 256, self.hidden_dim))

    def forward(self, x):
        """
        Forward pass of the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, #features, seq_len).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, #features, seq_len).
        """
        x = x.transpose(1, 2)
        if self.ReVIn:
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True)
            x = (x - mean) / (std + 0.001)
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out)
        if self.task == 'classification':
            tag_scores = torch.softmax(tag_space, axis=-1)
        elif self.ReVIn:
            tag_scores = tag_space * (std + 0.001) + mean
        else:
            tag_scores = tag_space
        return tag_scores.transpose(1, 2)

    def predict(self, x):
        """
        Predict function required by the user.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.forward(x)


if __name__ == "__main__":
    print("Running tests for FDTempExplainer...")
    print("Note: The user must have defined a predict function in the model.")

    # Specify devices
    cpu_device = "cpu"
    gpu_device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Simulate time series data in different formats
    n_samples, n_features, n_timesteps = 5, 2, 6
    numpy_data = np.random.rand(n_samples, n_features, n_timesteps)  # 5 samples, 2 features, 6 time steps 
    tensor_data = torch.from_numpy(numpy_data).float()

    # Load the trained model
    lstm = LSTM_new_1(input_dim=n_features, hidden_dim=120, output_dim=4, num_layers=3, task='classification')

    # Function to run explanation with specified parameters

    def run_explanation(data, only_last, device):
        explainer = FDTempExplainer(
            model=lstm.to(device)
        )      
        '''
        data: Input data to explain (array-like)
        only_last: For RNNs, if True, use only the result from the last time step for explanation.
                Otherwise, use results from all time steps. Default is True.
        patch_size: Size of the patches to use for attribution (must divide n_timesteps)
        sample_num: The number of samples used to compute the expected value.
        device: Target device for computation ("cpu" or "cuda:<gpu_id>")
        '''
        explainer.explain(
            data=data,
            only_last=only_last,
            patch_size=2,
            device=device
        )
        print(f"Data shape (only_last={only_last}, device={device}): {explainer.data.shape}")
        print(f"Main effects shape: {explainer.attribution_results['main_effect'].shape}")
        print(f"Interaction effects shape: {explainer.attribution_results['interaction_effect'].shape}")

    # Test cases
    print("\nTest Case 1: Data is NumPy array, only_last=True, device=CPU")
    run_explanation(numpy_data, only_last=True, device=cpu_device)

    print("\nTest Case 2: Data is NumPy array, only_last=False, device=CPU")
    run_explanation(numpy_data, only_last=False, device=cpu_device)

    print("\nTest Case 3: Data is Tensor, only_last=True, device=CPU")
    run_explanation(tensor_data, only_last=True, device=cpu_device)

    print("\nTest Case 4: Data is Tensor, only_last=False, device=CPU")
    run_explanation(tensor_data, only_last=False, device=cpu_device)

    if torch.cuda.is_available():
        print("\nTest Case 5: Data is NumPy array, only_last=True, device=GPU")
        run_explanation(numpy_data, only_last=True, device=gpu_device)

        print("\nTest Case 6: Data is NumPy array, only_last=False, device=GPU")
        run_explanation(numpy_data, only_last=False, device=gpu_device)

        print("\nTest Case 7: Data is Tensor, only_last=True, device=GPU")
        run_explanation(tensor_data, only_last=True, device=gpu_device)

        print("\nTest Case 8: Data is Tensor, only_last=False, device=GPU")
        run_explanation(tensor_data, only_last=False, device=gpu_device)

    print("Tests completed successfully!")