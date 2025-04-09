def create_app(*args, **kwargs):
    from mindxlib.visualization.interactive import create_app as _create_app
    return _create_app(*args, **kwargs)

def plot_static_gam(*args, **kwargs):
    from mindxlib.visualization.plots import plot_static_gam as _plot_static_gam
    return _plot_static_gam(*args, **kwargs)