from __future__ import print_function
from abc import ABC

import shap
from mindxlib.base import PostHocWhiteBoxBase, PostHocBlackBoxBase


class KernelExplainer(PostHocBlackBoxBase):
    """
    This class wraps the source class `KernelExplainer <https://shap.readthedocs.io/>`_
    available in the `SHAP <https://github.com/slundberg/shap>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(KernelExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.KernelExplainer(*argv, **kwargs)

    def set_parameters(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one ore more input instances.
        """
        return (self.explainer.shap_values(*argv, **kwargs))


class GradientExplainer(PostHocWhiteBoxBase):
    """
    This class wraps the source class `GradientExplainer <https://shap.readthedocs.io/>`_
    available in the `SHAP <https://github.com/slundberg/shap>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(GradientExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.GradientExplainer(*argv, **kwargs)

    def set_parameters(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        return (self.explainer.shap_values(*argv, **kwargs))


class DeepExplainer(PostHocWhiteBoxBase):
    """
    This class wraps the source class `DeepExplainer <https://shap.readthedocs.io/>`_
    available in the `SHAP <https://github.com/slundberg/shap>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(DeepExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.DeepExplainer(*argv, **kwargs)

    def set_parameters(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        return (self.explainer.shap_values(*argv, **kwargs))


class TreeExplainer(PostHocWhiteBoxBase):
    """
    This class wraps the source class `TreeExplainer <https://shap.readthedocs.io/>`_
    available in the `SHAP <https://github.com/slundberg/shap>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(TreeExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.TreeExplainer(*argv, **kwargs)

    def set_parameters(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        return (self.explainer.shap_values(*argv, **kwargs))


class LinearExplainer(PostHocWhiteBoxBase):
    """
    This class wraps the source class `Linearexplainer <https://shap.readthedocs.io/>`_
    available in the `SHAP <https://github.com/slundberg/shap>`_ library.
    Additional variables or functions from the source class can also be accessed via the 'explainer'
    object variable that is initialized in '__init__' function of this class.
    """

    def __init__(self, *argv, **kwargs):
        """
        Initialize shap kernelexplainer object.
        """
        super(LinearExplainer, self).__init__(*argv, **kwargs)

        self.explainer = shap.LinearExplainer(*argv, **kwargs)

    def set_parameters(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, *argv, **kwargs):
        """
        Explain one or more input instances.
        """
        return (self.explainer.shap_values(*argv, **kwargs))