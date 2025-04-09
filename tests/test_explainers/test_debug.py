
# import dash_bootstrap_components as dbc
# from dash import Dash, Input, Output, callback, html
# import mindxlib.visualization.dash_vis_components
# from mindxlib.visualization.dash_vis_components import CalHeatmap, LineChart, ShapeEnsemble, Waterfall
from mindxlib.visualization.interactive import create_app


# 在外部显式启动服务
if __name__ == "__main__":
    # import webbrowser
    # webbrowser.open("http://localhost:8050")
    create_app(None, None)