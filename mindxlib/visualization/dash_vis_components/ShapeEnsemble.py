# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class ShapeEnsemble(Component):
    """A ShapeEnsemble component.


Keyword arguments:

- id (string; required)

- choosedId (number; required)

- data (dict; required)

- feature_info (dict; required)

- hoveredFeature (string; optional)

- waterfallDataAll (list; required)"""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_vis_components'
    _type = 'ShapeEnsemble'
    @_explicitize_args
    def __init__(self, id=Component.REQUIRED, data=Component.REQUIRED, feature_info=Component.REQUIRED, choosedId=Component.REQUIRED, waterfallDataAll=Component.REQUIRED, hoveredFeature=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'choosedId', 'data', 'feature_info', 'hoveredFeature', 'waterfallDataAll']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'choosedId', 'data', 'feature_info', 'hoveredFeature', 'waterfallDataAll']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        for k in ['id', 'choosedId', 'data', 'feature_info', 'waterfallDataAll']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')

        super(ShapeEnsemble, self).__init__(**args)
