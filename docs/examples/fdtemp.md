# FDTemp (Functional Decomposition Temperature)

FDTemp is a method for explaining temporal black-box models through functional decomposition. It provides interpretable insights into how different features and their interactions contribute to model predictions over time.

## Class Definition
class mindxlib.explainer.FDTempExplainer(
    model: callable,
    data: np.ndarray = None,
    n_epochs: int = 100,
    batch_size: int = 1,
    lr: float = 0.01,
    layer_dim: list = [80, 64, 64, 32],
    device: str = "cpu",
    vae_train_print: bool = False,
    **kwargs
)

### Parameters

- **model** : `callable`  
  The model to explain (must implement predict() method)

- **data** : `np.ndarray`, default=None  
  Training data for VAE of shape (n_samples, n_features, n_timesteps)

- **n_epochs** : `int`, default=100  
  Number of training epochs for the VAE

- **batch_size** : `int`, default=1  
  Batch size for VAE training

- **lr** : `float`, default=0.01  
  Learning rate for VAE optimizer

- **layer_dim** : `list`, default=[80, 64, 64, 32]  
  Layer dimensions for VAE architecture [input_dim, hidden1, hidden2, latent_dim]

- **device** : `str`, default="cpu"  
  Computation device ("cpu" or "cuda:<gpu_id>")

- **vae_train_print** : `bool`, default=False  
  Whether to print VAE training progress

## Usage

```python
from mindxlib import FDTempExplainer

# Initialize the explainer
explainer = FDTempExplainer(model)

# Generate explanations
explanation = explainer.explain(X)

# Access main effects and interaction effects
main_effects = explanation.main_effect
interaction_effects = explanation.interaction_effect
```

## Methods

### explain()

```python
def explain(self,
           data_test=None,      
           only_last=True,      
           baseline=None,       
           patch_size=1,       
           sample_num=10,       
           **kwargs            
           )
```

**Parameters:**
- **data_test** : `numpy.ndarray`
  - Time series samples to explain
  - Shape: (n_samples, n_features, n_timesteps)
  - Must Input

- **only_last** : `bool`
  - For recurrent models only:
    - True: Explain only final timestep output
    - False: Explain all timesteps independently
  - Default: True

- **baseline** : `None`, `float`, or `numpy.ndarray`
  - Reference values for attribution
  - Default: None

- **patch_size** : `int`
  - Size of temporal segments for analysis:
    - Must evenly divide n_timesteps
    - Larger values capture longer-term patterns
    - Smaller values give finer-grained explanations
  - Default: 1

- **sample_num** : `int`
  - Number of samples used to compute the expected value.
  - Default: 10

- **kwargs** : `dict`
  - Additional explanation parameters

**Returns:**
- `FeatureImportanceExplanation` object containing decomposed feature effects:

1. **Main Effects** (Individual Contributions):
- `.feature_importance['main_effect']`: 
- Shape: `(n_samples, n_features, n_patches, output_dim, n_timesteps)`
- Description: 
    - Quantifies isolated impact of each feature patch
    - Positive/Negative values indicate directional influence

2. **Interaction Effects** (Pairwise Interactions):
- `.feature_importance['interaction_effect']`:
- Shape: `(n_samples, n_features, n_patches, n_patches, output_dim, n_timesteps)`  
- Description:
    - Captures non-linear feature interplay
    - Symmetric matrix (interaction[i,j] == interaction[j,i])

## Examples

```python
import numpy as np
import torch
from mindxlib.explainers.timeseries import FDTempExplainer

# 1. Prepare a simple LSTM model (must implement predict() method)
class TimeSeriesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=32)
        self.linear = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, timesteps) -> (batch, timesteps, features)
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :])  # Only use last timestep
    
    def predict(self, x):
        return self.forward(x)

# 2. Create synthetic data
train_data = torch.from_numpy(np.random.rand(8, 3, 20))  # 100 samples, 3 features, 20 timesteps
test_sample = torch.from_numpy(np.random.rand(1, 3, 20))   # Single sample to explain

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 3. Initialize and train explainer
explainer = FDTempExplainer(
    model=TimeSeriesModel().to(device),
    data=train_data,
    n_epochs=50,
    batch_size=16,
    device=device
)

# 4. Generate explanations
explanation = explainer.explain(
    data_test=test_sample,
    only_last=True,  # Explain final prediction
    patch_size=2,    # Analyze 2-timestep segments
    sample_num=20    # Use 20 samples for stable estimates
)

# 5. Interpret results
main_effects = explanation.attribution_results['main_effect']  # Shape: (1, 3, 10, 1, 1)
interactions = explanation.attribution_results['interaction_effect']  # Shape: (1, 3, 10, 10, 1, 1)

print("Main effects per feature segment:", main_effects.squeeze().shape)
print("Feature interactions:", interactions.squeeze().shape)

```

## References

Yang, L., Tong, Y., Gu, X., & Sun, L. (2024). Explain temporal black-box models via functional decomposition. In Forty-first International Conference on Machine Learning. 