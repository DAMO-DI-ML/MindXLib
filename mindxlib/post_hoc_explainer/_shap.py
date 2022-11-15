from __future__ import print_function
from abc import ABC

import numpy as np

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


class PermutationExplainer():
    """
    This method approximates the Shaply values by iterating through permutations of the inputs.
    """

    def __init__(self, model, n_permutations: int=1024, *argv, **kwargs):
        """
        Initialize shap permutationexplainer object.
        """
        # super(PermutationExplainer, self).__init__(*argv, **kwargs)

        # self.explainer = shap.LinearExplainer(*argv, **kwargs)
        self.model = model
        self.n_permutations = n_permutations


    def set_parameters(self, *argv, **kwargs):
        """
        Optionally, set parameters for the explainer.
        """
        pass

    def explain_instance(self, Xt, Xb, groups: np.ndarray, seed: int=(1<<31)-1, *argv, **kwargs):
        """
        Explain one or more input instances.

        Args:
            Xt: (M, D) array, targets to explain.
            Xb: (M, D) array, baselines.
            groups: (N, ) array of indices, group of feature to change together.

        Returns:
            contribs: (M, D) array.
        """
        self.rng = np.random.default_rng(seed)
        groups = groups + 1
        M, D = Xt.shape
        # assert D <= 20
        N = len(groups)
        P = self.n_permutations

        perms = np.stack([self.rng.permutation(N) for _ in range(P // 2)], axis=0)
        perms = np.concatenate((perms, perms[:, ::-1]), axis=0)

        index_perms = groups[perms]
        tril = np.tril(np.ones((N + 1, N), dtype=int), k=-1)
        partial = index_perms[:,np.newaxis,:]* tril # (P, D+1, D)
        
        S = P * (N + 1)
        partial = np.apply_along_axis(np.concatenate, 1, partial.reshape(S, N))
        masks = np.zeros((partial.shape[0],partial.shape[1]+1), dtype=float)
        np.put_along_axis(masks, partial, 1, axis=-1)
        masks = masks[:, np.newaxis, 1:]

        mixed = Xt[np.newaxis, ...] * masks + Xb[np.newaxis, ...] * (1 - masks)
        preds = self.model.predict(mixed.reshape(S * M, D)).reshape(P, N+1, M)

        gains = preds[:, 1:, :] - preds[:, :-1, :]  # (P, N, M)
        contribs = np.zeros_like(gains)
        np.put_along_axis(contribs, perms[..., np.newaxis], gains, axis=-2)
        contribs = np.mean(contribs, axis=0)
        return np.transpose(contribs)

    # def explain_instance(self, *argv, **kwargs):
    #     """
    #     Explain one or more input instances.
    #     """
    #     return (self.explainer.shap_values(*argv, **kwargs))
