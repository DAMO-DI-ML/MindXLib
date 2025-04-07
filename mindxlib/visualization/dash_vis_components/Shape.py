# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class Shape(Component):
    """A Shape component.


Keyword arguments:

- data (list; required)

- hoveredFeature (string; optional)

- instanceData (dict; required)

- xInfo (dict; required)"""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_vis_components'
    _type = 'Shape'
    @_explicitize_args
    def __init__(self, data=Component.REQUIRED, xInfo=Component.REQUIRED, instanceData=Component.REQUIRED, hoveredFeature=Component.UNDEFINED, **kwargs):
        self._prop_names = ['data', 'hoveredFeature', 'instanceData', 'xInfo']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['data', 'hoveredFeature', 'instanceData', 'xInfo']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        for k in ['data', 'instanceData', 'xInfo']:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')

        super(Shape, self).__init__(**args)
