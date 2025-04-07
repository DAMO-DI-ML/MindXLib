__version__ = "0.1.0"

def __getattr__(name):
    """Lazy import mechanism"""
    if name == 'SSRL':
        from mindxlib.explainers.rules.rulelist.rulelist_SSRL import SSRL
        return SSRL
    elif name == 'GAM':
        from mindxlib.explainers.interactive_gam.gam import GAM
        return GAM
    elif name == 'ShapExplainer':
        from mindxlib.explainers.common_library_attribution.shap import ShapExplainer
        return ShapExplainer
    elif name == 'Diver':
        from mindxlib.explainers.rules.ruleset import Diver
        return Diver
    elif name == 'DrillUp':
        from mindxlib.explainers.rules.ruleset import DrillUp
        return DrillUp
    elif name == 'RuleSetImb':
        from mindxlib.explainers.rules.ruleset import RuleSetImb
        return RuleSetImb
    elif name == 'RuleSet':
        from mindxlib.explainers.rules.ruleset import RuleSet
        return RuleSet
    elif name == 'FDTempExplainer':
        from mindxlib.explainers.timeseries import FDTempExplainer
        return FDTempExplainer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['SSRL', 'FDTempExplainer', 'Diver', 'DrillUp', 'RuleSetImb', 'RuleSet', 'GAM', 'ShapExplainer']