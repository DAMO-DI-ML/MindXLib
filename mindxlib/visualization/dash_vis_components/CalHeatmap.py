# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class CalHeatmap(Component):
    """A CalHeatmap component.


Keyword arguments:

- data (dict; default {      x_labels: [],      y_labels: [],     value: [],  })

    `data` is a dict with keys:

    - value (list of list of number | stringss; required)

    - x_labels (list of strings; required)

    - y_labels (list of strings; required)"""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_vis_components'
    _type = 'CalHeatmap'
    @_explicitize_args
    def __init__(self, data=Component.UNDEFINED, **kwargs):
        self._prop_names = ['data']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['data']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args}

        super(CalHeatmap, self).__init__(**args)
