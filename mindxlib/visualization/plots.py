import matplotlib.pyplot as plt
import numpy as np
from shap.plots import waterfall, bar

def plot_waterfall(explanation, **kwargs):
    """
    Plot waterfall chart for SHAP explanations
    
    Args:
        explanations: Single shapExplanation instance or list of shapExplanation instances
        index (int): Index to use when explanations is a list
        **kwargs: Additional arguments passed to shap.waterfall_plot
    """
    waterfall(explanation.shap_explanation[:,:,0][0], **kwargs)


def plot_static_gam(model, feature_indices=None, figsize=(12, 10), display=True, 
             title=None, xlabel=None, ylabel="Attribution", show_density=True, 
             color='#1f77b4', linestyle='-', linewidth=2, alpha=0.7, 
             density_color='#ff7f0e', density_alpha=0.3, density_markersize=5,
             use_color_cycle=False, save_path=None, dpi=300, xlim=None, ylim=None, 
             layout=None, **kwargs):
        """
        Plot the shape functions for the specified features.
        
        Parameters
        ----------
        feature_indices : int, str, or list, optional
            Indices or names of features to plot. If None, all features are plotted.
            Can be a single integer index, a single string feature name, or a list 
            containing a mix of integer indices and string feature names.
        figsize : tuple, default=(12, 10)
            Figure size.
        display : bool, default=True
            Whether to display the figure immediately using plt.show().
        title : str or list of str, optional
            Title for the plot or list of titles for each subplot.
        xlabel : str or list of str, optional
            Label for x-axis or list of labels for each subplot. If None, feature names are used.
        ylabel : str, default="Attribution"
            Label for y-axis.
        show_density : bool, default=True
            Whether to show density of data points as a rug plot.
        color : str or list, default='#1f77b4'
            Color for the line or list of colors for each subplot. By default, all plots use the same blue color.
        linestyle : str or list, default='-'
            Line style or list of line styles for each subplot.
        linewidth : float or list, default=2
            Line width or list of line widths for each subplot.
        alpha : float, default=0.7
            Alpha transparency for the line plot.
        density_color : str or list, default='#ff7f0e'
            Color for the density plot. By default, all density plots use the same orange color.
        density_alpha : float, default=0.3
            Alpha transparency for the density plot.
        density_markersize : float, default=5
            Size of markers in the density plot.
        use_color_cycle : bool, default=False
            If True, uses matplotlib's default color cycle for lines instead of a single color.
        save_path : str, optional
            Path to save the figure. If provided, the figure will be saved to this location.
            The file format is determined by the file extension (e.g., .png, .pdf, .svg).
        dpi : int, default=300
            Resolution of the saved figure in dots per inch.
        xlim : tuple or list of tuples, optional
            The x limits (min, max) for the plot or a list of tuples for each subplot.
        ylim : tuple or list of tuples, optional
            The y limits (min, max) for the plot or a list of tuples for each subplot.
        layout : tuple, optional
            The layout of subplots as (rows, cols). If None, a square-ish layout is used.
        **kwargs : dict
            Additional keyword arguments to pass to matplotlib.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        if model.sfo is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Process feature indices
        if feature_indices is None:
            # Use all features
            indices = list(range(len(model.feature_names)))
        else:
            # Convert to list if a single index/name is provided
            if isinstance(feature_indices, (int, str)):
                feature_indices = [feature_indices]
            
            # Convert any feature names to indices
            indices = []
            for idx in feature_indices:
                if isinstance(idx, str):
                    if idx in model.feature_names:
                        indices.append(model.feature_names.index(idx))
                elif isinstance(idx, int):
                    if 0 <= idx < len(model.feature_names):
                        indices.append(idx)
                else:
                    raise ValueError(f"Feature identifier must be a string or integer, got {type(idx)}")
        
        # Create a figure with the specified size
        fig = plt.figure(figsize=figsize)
        
        # Prepare plot parameters
        n_plots = len(indices)
        
        # Determine subplot layout
        if layout is None:
            # Default to square-ish layout
            M = int(round(np.sqrt(n_plots)))
            N = int(np.ceil(n_plots / M))
        else:
            # Use specified layout
            M, N = layout
            if M * N < n_plots:
                print(f"Warning: Layout {layout} can only fit {M*N} plots, but {n_plots} were requested.")
                # Adjust to fit all plots
                N = int(np.ceil(n_plots / M))
        
        # Handle list or single value for plot parameters
        def ensure_list(param, n):
            if isinstance(param, list):
                return param
            else:
                return [param] * n
        
        # Use matplotlib's default color cycle if requested
        if use_color_cycle:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            # Repeat the color cycle if needed
            colors = [colors[i % len(colors)] for i in range(n_plots)]
        else:
            colors = ensure_list(color, n_plots)
        
        linestyles = ensure_list(linestyle, n_plots)
        linewidths = ensure_list(linewidth, n_plots)
        
        # Handle density colors
        density_colors = ensure_list(density_color, n_plots)
        
        # Handle titles and xlabels
        if title is not None:
            titles = ensure_list(title, n_plots)
        else:
            titles = [None] * n_plots
        
        if xlabel is not None:
            xlabels = ensure_list(xlabel, n_plots)
        else:
            xlabels = [model.feature_names[i] for i in indices]
        
        # Handle axis limits
        if xlim is not None:
            xlims = ensure_list(xlim, n_plots)
        else:
            xlims = [None] * n_plots
        
        if ylim is not None:
            ylims = ensure_list(ylim, n_plots)
        else:
            ylims = [None] * n_plots
        
        # Create subplots
        for i, (idx, col_index) in enumerate(enumerate(indices)):
            if i < M * N:  # Only create plots that fit in the layout
                ax = plt.subplot(M, N, i+1)
                
                # Get data for the feature
                sort_index = np.argsort(model.model.X[:, col_index])
                x_mark = model.model.X[:, col_index][sort_index]
                y_mark = model.model.shapeFunctionOptimizerList[col_index].predict(x_mark)
                
                # Rescale data
                x_mark = model.model._rescale_data(x_mark, idx=col_index)
                y_mark = y_mark * model.model.scale_info['y_scale'] + model.model.scale_info['y_offset'] / len(model.sfo)
                
                # Plot shape function
                ax.plot(x_mark, y_mark, color=colors[i], linestyle=linestyles[i], 
                        linewidth=linewidths[i], alpha=alpha, **kwargs)
                
                # Show density if requested
                if show_density:
                    # Get original data for density plot
                    x_orig = model.model._rescale_data(model.model.X[:, col_index], idx=col_index)
                    
                    # Add rug plot at the bottom
                    baseline = min(y_mark) - 0.1 * (max(y_mark) - min(y_mark))
                    ax.plot(x_orig, np.ones_like(x_orig) * baseline, '|', 
                            color=density_colors[i], alpha=density_alpha, markersize=density_markersize)
                
                # Set labels and title
                ax.set_xlabel(xlabels[i])
                ax.set_ylabel(ylabel)
                if titles[i]:
                    ax.set_title(titles[i])
                
                # Set axis limits if provided
                if xlims[i] is not None:
                    ax.set_xlim(xlims[i])
                if ylims[i] is not None:
                    ax.set_ylim(ylims[i])
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        # Display the figure if requested
        if display:
            plt.show()
        
        # Return the figure
        return fig