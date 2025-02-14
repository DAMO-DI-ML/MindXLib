import numpy as np
from ...core.base import WhiteBoxBase

class IntegratedGradients(WhiteBoxBase):
    """
    Integrated Gradients attribution method for deep learning models.
    Based on the paper: https://arxiv.org/abs/1703.01365
    """
    
    def __init__(self, model, baseline=None, steps=50, method="gausslegendre"):
        """
        Args:
            model: Model object that has predict_proba and gradient methods
            baseline: Baseline input (e.g. zero image/black image/random noise)
                     If None, zero array will be used
            steps: Number of steps for path integral approximation
            method: Path integral approximation method, one of:
                   - 'gausslegendre': Gauss-Legendre quadrature
                   - 'riemann': Riemann sum approximation
        """
        super().__init__(model)
        self.baseline = baseline
        self.steps = steps
        self.method = method

    def _get_gradients(self, inputs, target_labels=None):
        """
        Get gradients of model output with respect to inputs
        """
        if hasattr(self.model, 'gradient'):
            return self.model.gradient(inputs, target_labels)
        else:
            raise ValueError("Model must implement gradient() method")

    def _get_integral_approximation(self, input_tensor, baseline, target_label):
        """
        Approximates the path integral using the specified method
        """
        if self.method == "gausslegendre":
            # Use Gauss-Legendre quadrature for better approximation
            alphas, weights = np.polynomial.legendre.leggauss(self.steps)
            alphas = alphas * 0.5 + 0.5  # Scale from [-1, 1] to [0, 1]
            weights = weights * 0.5  # Scale weights accordingly
        else:
            # Use simple Riemann sum
            alphas = np.linspace(0, 1, self.steps)
            weights = np.ones_like(alphas) / self.steps

        # Compute interpolated inputs
        interpolated = np.zeros((self.steps,) + input_tensor.shape[1:])
        for i, alpha in enumerate(alphas):
            interpolated[i] = baseline + alpha * (input_tensor - baseline)

        # Get gradients for all interpolated inputs
        gradients = self._get_gradients(interpolated, target_label)

        # Compute integral approximation
        integrated_gradients = np.zeros_like(input_tensor)
        for i, weight in enumerate(weights):
            integrated_gradients += weight * gradients[i]

        return integrated_gradients * (input_tensor - baseline)

    def explain_instance(self, input_tensor, target_label=None, baseline=None):
        """
        Explains the prediction for a given input tensor.
        
        Args:
            input_tensor: Input to explain (numpy array)
            target_label: Target class to explain (for classification)
            baseline: Optional baseline input (if None, uses self.baseline)
            
        Returns:
            attributions: Attribution scores for each input feature
        """
        if baseline is None:
            baseline = self.baseline
        if baseline is None:
            baseline = np.zeros_like(input_tensor)

        # Expand dimensions if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.reshape(1, -1)
            baseline = baseline.reshape(1, -1)

        # Compute integrated gradients
        attributions = self._get_integral_approximation(
            input_tensor, baseline, target_label
        )

        return attributions

    def explain_batch(self, inputs, target_labels=None, baselines=None):
        """
        Explains predictions for a batch of inputs
        
        Args:
            inputs: Batch of inputs to explain
            target_labels: Target classes to explain
            baselines: Optional baselines for each input
            
        Returns:
            batch_attributions: Attribution scores for each input
        """
        if baselines is None:
            baselines = [self.baseline] * len(inputs)
        if target_labels is None:
            target_labels = [None] * len(inputs)

        batch_attributions = []
        for input_tensor, baseline, target in zip(inputs, baselines, target_labels):
            attribution = self.explain_instance(input_tensor, target, baseline)
            batch_attributions.append(attribution)

        return np.array(batch_attributions)

    def validate_attributions(self, attributions, inputs, baseline=None):
        """
        Validates attributions using the completeness axiom:
        sum of attributions should approximate f(x) - f(baseline)
        
        Args:
            attributions: Attribution scores
            inputs: Original inputs
            baseline: Baseline inputs
            
        Returns:
            completeness_score: How well attributions satisfy completeness
        """
        if baseline is None:
            baseline = self.baseline
        if baseline is None:
            baseline = np.zeros_like(inputs)

        input_predictions = self.model.predict_proba(inputs)
        baseline_predictions = self.model.predict_proba(baseline)
        
        attribution_sum = np.sum(attributions, axis=1)
        prediction_diff = input_predictions - baseline_predictions
        
        completeness_score = np.mean(
            np.abs(attribution_sum - prediction_diff) / (np.abs(prediction_diff) + 1e-7)
        )
        
        return completeness_score
