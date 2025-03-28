# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class Waterfall(Component):
    """A Waterfall component.


Keyword arguments:

- id (string; required)

- choosedId (number; required)

- data (list of dicts; required)


- y_axis_label (string; optional)

- feature_info (dict; optional)

    `feature_info` is a dict with keys:


- hoveredFeatureIndex (string; optional)

- intercept (number; required)"""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_vis_components'
    _type = 'Waterfall'
    @_explicitize_args
    def __init__(self, id=Component.REQUIRED, data=Component.REQUIRED, choosedId=Component.REQUIRED, intercept=Component.REQUIRED, feature_info=Component.UNDEFINED, hoveredFeatureIndex=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'choosedId', 'data', 'feature_info', 'hoveredFeatureIndex', 'intercept']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'choosedId', 'data', 'feature_info', 'hoveredFeatureIndex', 'intercept']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        for k in ['id', 'choosedId', 'data', 'intercept']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')

        super(Waterfall, self).__init__(**args)
