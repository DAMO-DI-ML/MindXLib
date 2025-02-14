from lime import lime_image, lime_text, lime_tabular
from ...core.base import BlackBoxBase

class LimeExplainer(BlackBoxBase):
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
    pass
