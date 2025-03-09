import numpy as np
from mindxlib.base.explainer import FeatureImportanceExplainer
from mindxlib.base.explanation import FeatureImportanceExplanation

class IntegratedGradients(FeatureImportanceExplainer):
    """
    Integrated Gradients attribution method for deep learning models.
    Based on the paper: https://arxiv.org/abs/1703.01365
    """
    
    def __init__(self, model, steps=50, method="gausslegendre"):
        """
        Args:
            model: Model object that has predict_proba and gradient methods
            steps: Number of steps for path integral approximation
            method: Path integral approximation method, one of:
                   - 'gausslegendre': Gauss-Legendre quadrature
                   - 'riemann': Riemann sum approximation
        """
        super().__init__(model)
        self.steps = steps
        self.method = method

    def _initial_baseline(self, inputs, baseline):
        """
        Initialize the baseline for the Integrated Gradients method.
        
        Args:
            inputs (numpy.ndarray): Input data to be explained.
            baseline (numpy.ndarray or None): User-provided baseline input. If None, a zero baseline is used.
            
        Returns:
            numpy.ndarray: The initialized baseline.
            
        Raises:
            ValueError: If the provided baseline shape is not compatible with the input shape.
        """
        if baseline is None:
            return np.zeros_like(inputs)
        else:
            if baseline.shape == inputs.shape:
                return baseline
            elif len(baseline.shape) == len(inputs.shape) - 1:
                return baseline * len(inputs)
            else:
                raise ValueError("IG can only have one baseline at a time")

    def _get_gradients(self, inputs):
        """
        Get gradients of model output with respect to inputs
        """
        if hasattr(self.model, 'gradient'):
            return self.model.gradient(inputs)
        else:
            raise ValueError("Model must implement gradient() method")

    def _get_integral_approximation(self, input_tensor, baseline):
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
        gradients = self._get_gradients(interpolated)

        # Compute integral approximation
        integrated_gradients = np.zeros(gradients.shape[1:], dtype=np.float64)
        for i, weight in enumerate(weights):
            integrated_gradients += weight * gradients[i]

        return integrated_gradients * (input_tensor - baseline).reshape(-1, 1)

    def explain_instance(self, input_tensor, baseline=None):
        """
        Explains the prediction for a given input tensor.
        
        Args:
            input_tensor: Input to explain (numpy array)
            baseline: Optional baseline input 
            
        Returns:
            attributions: Attribution scores for each input feature
        """

        # Expand dimensions if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.reshape(1, -1)
            baseline = baseline.reshape(1, -1)

        # Compute integrated gradients
        attributions = self._get_integral_approximation(
            input_tensor, baseline
        )
        return attributions

    def _compute_attributions(self, inputs, baselines=None):
        """
        Explains predictions for a batch of inputs
        
        Args:
            inputs: Batch of inputs to explain
            baselines: Optional baselines for each input
            
        Returns:
            batch_attributions: Attribution scores for each input
        """

        batch_attributions = []
        for input_tensor, baseline in zip(inputs, baselines):
            attribution = self.explain_instance(input_tensor, baseline)
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
