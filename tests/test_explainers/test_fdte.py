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
    print("sys.path", sys.path)

    # Specify device
    device = "cuda:1"

    # Simulate time series data (can be either np.array or torch.tensor)
    data = np.random.rand(5, 50, 2)  # 100 samples, 50 time steps, 2 features
    # data = torch.from_numpy(data)

    # Load the trained model
    lstm = LSTM_new_1(input_dim=1, hidden_dim=120, output_dim=4, num_layers=3, task='classification').to(device)
    lstm.load_state_dict(torch.load('/mnt/workspace/workgroup/workgroup/yitian/MindXLib/tests/test_explainers/test_model/lstm-LKA.pt'))

    # Instantiate FDTempExplainer
    explainer = FDTempExplainer(
        model=lstm,
        data=data
    )

    # Call the explain method for explanation (for RNN models, only use the last time step's output)
    explainer.explain(
        args='only_last',
        patch_size=5,
        test_size=0.2,
        random_state=42,
        device=device
    )

    # Call the explain method for explanation (for RNN models, use all time steps' output)
    # explainer.explain(
    #     args='None',
    #     patch_size=5,
    #     test_size=0.2,
    #     random_state=42,
    #     device=device
    # )

    # Print the shape of the dataset used for explanation
    print("Data shape:", explainer.data.shape)

    # Get main effects and interaction effects
    print("Main effects shape:", explainer.attribution_results['main_effect'].shape)
    print("Interaction effects shape:", explainer.attribution_results['interaction_effect'].shape)

    print("Tests completed successfully!")