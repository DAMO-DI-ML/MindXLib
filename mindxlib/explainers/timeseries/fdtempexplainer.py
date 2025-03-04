from mindxlib.base.explainer import FeatureImportanceExplainer
from mindxlib.base.explanation import FeatureImportanceExplanation
from .explain_utils import ImpVAE, PatchAttributionTorch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch 

class FDTempExplainer(FeatureImportanceExplainer):
    """Feature Decomposition Temperature explainer for time series data"""
    
    def __init__(self, model, data=None, **kwargs):
        """Initialize FDTemp explainer
        
        Args:
            model: The model to explain
            data: Optional input data for initialization (array-like)
                For time series: shape (n_samples, n_features, n_timesteps)
            **kwargs: Additional arguments for specific explainers
        """
        super().__init__(model, data, **kwargs)
        self._explanation = None
        self.model = model

    def _move_data_to_device(self, data, device="cpu"):
        """Move data to the specified device (CPU or GPU).
        
        Args:
            data: Input data (array-like)
            device: Target device ("cpu" or "cuda:<gpu_id>")
            
        Returns:
            Moved data on the specified device.
        """
        # Convert data to a PyTorch tensor if it's not already
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()  # Convert NumPy array to PyTorch tensor
        
        # Validate and process the device argument
        if device == "cpu":
            target_device = torch.device("cpu")
        elif device.startswith("cuda:"):
            if not torch.cuda.is_available():
                print("No GPU available. Using CPU instead.")
                target_device = torch.device("cpu")
            else:
                try:
                    gpu_id = int(device.split(":")[1])  # Extract GPU ID
                    target_device = torch.device(f"cuda:{gpu_id}")  # Create a device object
                except ValueError:
                    print(f"Invalid GPU ID in device specification: {device}. Using CPU instead.")
                    target_device = torch.device("cpu")
        else:
            print(f"Unsupported device specification: {device}. Using CPU instead.")
            target_device = torch.device("cpu")
        
        data = data.to(target_device)  # Move data to the specified device
        return data

    def explain(self, data=None, only_last=True, baseline=None, patch_size=1, sample_num = 10, device="cpu", **kwargs):
        """Generate feature importance explanations
        
        Args:
            data: Input data to explain (array-like)
            only_last: For RNNs, if True, use only the result from the last time step for explanation.
                    Otherwise, use results from all time steps.
            baseline: Optional reference values for computing feature importance
                    Default is None, in which case method-specific defaults are used
            patch_size: Size of the patches to use for attribution (must divide n_timesteps)
            sample_num: The number of samples used to compute the expected value.
            device: Target device for computation ("cpu" or "cuda:<gpu_id>")
            **kwargs: Additional explanation parameters
            
        Returns:
            self: The explainer instance with computed explanations
        """
        # Set the device for computation
        self.device = device

        # Validate data format and patch size
        self.data = self._validate_data(data, patch_size) 

        # Move data to the specified device
        self.data = self._move_data_to_device(self.data, self.device)

        # Compute attributions
        self.attribution_results = self._compute_attributions(self.data, patch_size, sample_num, only_last, **kwargs)

        # Store the explanation
        self._explanation = FeatureImportanceExplanation(
            feature_importance=self.attribution_results
        )
        
        return self

    def _validate_data(self, data, patch_size):
        """Validate and format input data
        
        Args:
            data: Input data (array-like), can be a NumPy array, a PyTorch tensor, or a Pandas DataFrame
            patch_size: Size of the patches to use for attribution (must divide n_timesteps)
            
        Returns:
            Formatted data as numpy array
        """

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()  # Convert to NumPy array if it's a PyTorch tensor

        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array, a PyTorch tensor")

        if len(data.shape) != 3:
            raise ValueError("Input data must be 3D with shape (n_samples, n_features, n_timesteps)")

        _, _, n_timesteps = data.shape
        if n_timesteps % patch_size != 0:
            raise ValueError(f"patch_size ({patch_size}) must divide n_timesteps ({n_timesteps}) evenly")
        
        return data


    def _compute_attributions(self, data, patch_size, sample_num, only_last, **kwargs):
        """Compute feature attributions3
        
        Args:
            data: Time series data of shape (n_samples, n_features, n_timesteps) 
            patch_size: Size of the patches to use for attribution
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
                main_effect: Individual feature contributions, whose shape is [n_samples, n_features, patch_num, model_outdim, n_timesteps]
                interaction_effect: Higher-order interaction effects, whose shape is [n_samples, n_features, patch_num, patch_num, model_outdim, n_timesteps]
        """
        n_samples, n_features , n_timesteps = data.shape
        

        # Initialize arrays to store contribution values
        main_effects = None
        interaction_effects = None

        # Placeholder for generator and explainer initialization
    
        generator = ImpVAE(num_features=n_features, seq_len=n_timesteps, device=self.device, layer_dim=[80, 64, 64, 32], BN_enable=True).to(self.device)
        explainer = PatchAttributionTorch(
            func=self.model.predict,
            patch_size=patch_size,
            x_size=n_timesteps,
            is_numpy_model=False,
            generator=generator,
            only_last = only_last,
            sample_num=sample_num,
            lambda_1=0,
            kk=100,
            device=self.device
        )

        all_samples_main_effects = []
        all_samples_interaction_effects = []

        for sample in range(n_samples):
            sample_main_effects = []  # Store current sample's feature contributions
            sample_interaction_effects = []
            for feature in range(n_features):
                # Extract current sample and feature data
                explained_x = data[sample, :, :]
               
                # Compute contributions
                main_effect = explainer.attribute(explained_x, cared_fid=feature, lambda_1=0)
                interaction_effect = explainer.interaction_matrix.cpu().numpy()
                
                # Append contributions to current sample's lists
                sample_main_effects.append(main_effect)
                sample_interaction_effects.append(interaction_effect)
            
            # Stack current sample's contributions into an array
            all_samples_main_effects.append(np.stack(sample_main_effects, axis=0))
            all_samples_interaction_effects.append(np.stack(sample_interaction_effects, axis=0))

        # Stack all samples' contributions into an array
        all_samples_main_effects = np.stack(all_samples_main_effects, axis=0)
        all_samples_interaction_effects = np.stack(all_samples_interaction_effects, axis=0)

        main_effects = all_samples_main_effects
        interaction_effects = all_samples_interaction_effects

        return {
            'main_effect': main_effects,
            'interaction_effect': interaction_effects
        }

