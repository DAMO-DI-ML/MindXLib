# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class LineChart(Component):
    """A LineChart component.


Keyword arguments:

- data (list of dicts; required)"""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_vis_components'
    _type = 'LineChart'
    @_explicitize_args
    def __init__(self, data=Component.REQUIRED, **kwargs):
        self._prop_names = ['data']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['data']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        for k in ['data']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')

        super(LineChart, self).__init__(**args)
