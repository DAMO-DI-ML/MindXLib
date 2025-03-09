import numpy as np
from lime import lime_image, lime_text, lime_tabular
from mindxlib.base.explainer import FeatureImportanceExplainer

class LimeExplainer(FeatureImportanceExplainer):
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)

class LimeTextExplainer(LimeExplainer):
    def __init__(self, model, *argv, **kwargs):
        super().__init__(model)
        self.explainer = lime_text.LimeTextExplainer(*argv, **kwargs)

    def explain_instance(self, *argv, **kwargs):
        return self.explainer.explain_instance(*argv, **kwargs)

class LimeImageExplainer(LimeExplainer):
    pass

class LimeTabularExplainer(LimeExplainer):
    def __init__(self, model, *argv, **kwargs):
        """
        Initialize lime Tabular Explainer object
        """
        super(LimeTabularExplainer, self).__init__(model)
    
    def _initial_baseline(self, data, baseline):

        return baseline
    
    def _compute_attributions(self, datas, baseline, *argv, **kwargs):
        explainer = lime_tabular.LimeTabularExplainer(training_data=baseline, *argv, **kwargs)

        batch_attributions = []
        for data in datas:
            attribution = explainer.explain_instance(data, self.model)
            batch_attributions.append(attribution)
        
        return np.array(batch_attributions)


