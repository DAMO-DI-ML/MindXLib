import sys
import json
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, html
import importlib.util
import subprocess
import os
import threading
import webbrowser

# First, check if dash_vis_components is installed in the environment
try:
    # Try to import from the installed package in the environment
    import dash_vis_components
    from dash_vis_components import CalHeatmap, LineChart, ShapeEnsemble, Waterfall
    print("dash_vis_components found in environment.")
except ImportError:
    # If not found in environment, try to import directly from local path
    print("dash_vis_components not found in environment. Trying local import...")
    try:
        # Add the local package path to sys.path
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vis_components_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dash_vis_components')
        
        if os.path.exists(vis_components_dir):
            # Add the component directory to the Python path
            if vis_components_dir not in sys.path:
                sys.path.insert(0, os.path.dirname(vis_components_dir))
            
            # Import from local path
            from mindxlib.visualization.dash_vis_components import CalHeatmap, LineChart, ShapeEnsemble, Waterfall
            print(f"Successfully imported dash_vis_components from local path: {vis_components_dir}")
        else:
            raise ImportError(f"dash_vis_components directory not found at {vis_components_dir}")
            
    except Exception as e:
        print("Failed to import dash_vis_components: {}".format(e))
        print("Please install dash_vis_components manually or ensure it's in the correct location.")
        print("You can install it using pip: pip install dash-vis-components")
        sys.exit(1)

import numpy as np
import pandas as pd
import threading
import webbrowser


def load_data(model, data, intercept=False):
    """
    Convert data to the format required by visualization components.
    
    Parameters
    ----------
    model : GAM
        The fitted GAM model
    data : pandas.DataFrame or numpy.ndarray
        Data to visualize.
        
    Returns
    -------
    tuple
        (zip_data, data_dict, data_waterfall) formatted for visualization
    """
    if isinstance(data, pd.DataFrame):
        data = data[model.feature_names].to_numpy()
        
    # Get shape functions and confidence intervals
    shape_functions = model.get_shape_functions(intercept=intercept)
    # point_confidence_intervals = model.get_confidence_intervals(data)
    shape_confidence_intervals = model.get_shape_function_confidence_intervals(intercept=intercept)
    feature_names = model.feature_names
    
    # Initialize data structures
    zip_data = {}
    
    # For each feature, create the required data format for zip_data
    for feature_name in feature_names:
        x_values, y_values = shape_functions[feature_name]
        sort_idx = np.argsort(x_values)
        x_ci, lower_ci, upper_ci = shape_confidence_intervals[feature_name]
        
        # Create the data structure for ShapeEnsemble
        feature_data = []
        for i in range(len(x_values)):
            feature_data.append({
                "x": float(x_values[sort_idx][i]),
                "y": float(y_values[sort_idx][i]),
                "c": [float(lower_ci[sort_idx][i]), float(upper_ci[sort_idx][i])]  # Use confidence interval bounds
            })
        
        zip_data[feature_name] = feature_data
    
    # Create data_dict with the required structure
    data_dict = {
        "intercept": float(model.model.scale_info['y_offset']) if intercept else 0,
        "r2": 0.95,
        "feature_info": {}
    }
    
    # Add feature contributions to data_dict
    for feature_name in feature_names:
        data_dict['feature_info'][feature_name] = {
            "name": feature_name,
            "display": feature_name,
            "type": "numeric"
        }
    
    # Create data_waterfall for each row in the provided data
    data_waterfall = []
    predictions = model.predict(data)
    shape_predictions = model.get_shape_predictions(data, intercept=intercept)
    
    for idx in range(len(data)):
        instance = {
            "id": idx,
            "y": 0,
            "pred_y": float(predictions[idx]),
            "data": []
        }
        for feature_name in feature_names:
            # lower_bound, upper_bound = point_confidence_intervals[feature_name]
            instance["data"].append({
                "fea_idx": feature_name,
                "fea_val": float(data[idx, feature_names.index(feature_name)]),
                "pdep": float(shape_predictions[feature_name][idx]),
                "confi_u_X": float(shape_predictions[feature_name][idx]),
                "confi_l_X": float(shape_predictions[feature_name][idx])
            })
        data_waterfall.append(instance)
    
    return zip_data, data_dict, data_waterfall

def create_app(model, data, index=0, intercept = False, waterfall_height="40vh", port=8050, auto_open=True):
    """
    Create and run a Dash application for interactive visualization of the GAM model.
    
    Parameters
    ----------
    model : GAM
        The fitted GAM model
    data : pandas.DataFrame or numpy.ndarray
        Data to visualize
    index : int, default=0
        Initial index to display
    waterfall_height : str, default="40vh"
        Height of the waterfall plot
    port : int, default=8050
        Port to run the application on
    auto_open : bool, default=True
        Whether to automatically open the browser
    """
    if model is not None:
        zip_data, model_info, data_waterfall = load_data(model, data, intercept)
    else:
        zip_data = {'x1': [{'x': 1, 'y': 1, 'c': [1, 1]}, 
                        {'x': 2, 'y': 2, 'c': [2, 2]}],
                    'x2': [{'x': 1, 'y': 2, 'c': [2, 2]}, 
                        {'x': 2, 'y': 4, 'c': [4, 4]}]}
        data_waterfall = [{'id': 0, 'y': 0, 'pred_y': 3, 'data': [{'fea_idx': 'x1', 'fea_val': 1, 'pdep': 1, 'confi_u_X': 1, 'confi_l_X': 1}, 
                                                                    {'fea_idx': 'x2', 'fea_val': 1, 'pdep': 2, 'confi_u_X': 2, 'confi_l_X': 2}]},
                        {'id':1, 'y': 0, 'pred_y': 5, 'data': [{'fea_idx': 'x1', 'fea_val': 1, 'pdep': 1, 'confi_u_X': 1, 'confi_l_X': 1}, 
                                                                    {'fea_idx': 'x2', 'fea_val': 2, 'pdep': 4, 'confi_u_X': 4, 'confi_l_X': 4}]}]
        model_info = {'intercept': 0, 'r2': 0.95, 'feature_info': {'x1': {'name': 'x1', 'display': 'x1', 'type': 'numeric'}, 
                                                                    'x2': {'name': 'x2', 'display': 'x2', 'type': 'numeric'}}}
        
    # Get the path to the assets folder relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_folder = os.path.join(current_dir, "assets")
    print("Assets folder path:", assets_folder) 
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=assets_folder,
    )
        
    app.layout = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    # 在 Dash 中，组件的属性（props）是静态的，不能像在 React 中那样直接传递函数。
                    Waterfall(
                        id="waterfall-component",
                        data=data_waterfall,
                        choosedId=index,
                        intercept=model_info["intercept"],
                        feature_info=model_info["feature_info"],
                        hoveredFeatureIndex=None,
                        # setHoveredFeature=lambda *args: None,
                    ),
                    style={"height": waterfall_height},
                )
            ),
            html.Div(id="hovered-feature"),
            dbc.Row(
                dbc.Col(
                    ShapeEnsemble(
                        id="shape-ensemble-component",
                        data=zip_data,
                        feature_info=model_info["feature_info"],
                        choosedId=0,
                        waterfallDataAll=data_waterfall,
                        hoveredFeature=None,
                    ),
                )
            ),
        ]
    )

    @app.callback(
        Output("shape-ensemble-component", "hoveredFeature"),
        Input("waterfall-component", "hoveredFeatureIndex"),
    )
    def update_hovered_feature_index(hoveredFeatureIndex):
        return hoveredFeatureIndex
    
    if auto_open:
        # Open the browser after a short delay to ensure server has started
        timer = threading.Timer(0.5, lambda: webbrowser.open_new(f"http://127.0.0.1:{port}/"))
        timer.start()
    
    app.run(debug=False, port=port)
    
    return app

if __name__ == "__main__":
    create_app(None, None, index=1)
