import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# (A) UTILITIES FOR ALLUVIAL PLOT
###############################################################################
def sigmoid(x):
    """Sigmoid function used for smoothing ribbon curves."""
    return 1 / (1 + np.exp(-x))

def calc_sigmoid_line(width, y_start, y_end):
    """
    Returns arrays (xs, ys_under) that define the "lower boundary" of a sigmoid
    curve going from (x=0, y=y_start) to (x=width, y=y_end).
    """
    xs = np.linspace(-5, 5, num=50)
    ys = sigmoid(xs)
    xs = xs / 10 + 0.5  # rescale into [0..1]
    xs *= width
    ys = y_start + (y_end - y_start) * ys
    return xs, ys

###############################################################################
# (B) MAIN PLOT FUNCTION
###############################################################################
def plot(
    df, 
    xaxis_names,        # e.g. ["behavior", "baseline_bin", "day7_bin", "day30_bin"]
    y_name,             # numeric column for flow size, e.g. "freq"
    alluvium=None,      # column to color by, e.g. "behavior" or "sign"
    order_dict=None,    # dict specifying the order for each pillar (if desired)
    ignore_continuity=False,
    cmap_name='tab10',
    figsize=(6.4, 4.8)
):
    """
    Plots an alluvial diagram with multiple pillars (given by xaxis_names).
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have:
        - One column per "pillar" in xaxis_names (categorical or string)
        - A numeric column y_name for flow size
        - (Optional) 'alluvium' column to color flows
    xaxis_names : list of str
        The pillar columns in the order you want them displayed
    y_name : str
        Numeric column name for flow height/width
    alluvium : str or None
        If not None, each unique value in this column is assigned a unique color
    order_dict : dict or None
        If provided, you can specify an order for each pillar. 
        Example: {"baseline_bin": ["3-4","4-5","5-6",...]}
    ignore_continuity : bool
        If True, it draws each pair of pillars separately (not recommended).
        If False, it tries to draw them in one pass, connecting the pillars.
    cmap_name : str
        A matplotlib colormap name, e.g. "tab10"
    figsize : tuple
        Size of the figure

    Returns
    -------
    fig : matplotlib Figure
    """
    df = df.copy()
    
    # 1) Build stratum sums for each pillar
    stratum_dict = {}
    for xaxis in xaxis_names:
        # groupby that axis, sum up the numeric y_name
        # observed=False -> includes all categories that appear in data
        stratum_dict[xaxis] = df.groupby(xaxis, observed=False)[y_name].sum()
    
    # 2) If we have an order_dict, reorder existing categories (but don't force new)
    if order_dict:
        for key, orders in order_dict.items():
            if key in stratum_dict:
                # restrict to the intersection so we skip categories not in the data
                existing = stratum_dict[key].index.intersection(orders)
                stratum_dict[key] = stratum_dict[key].loc[existing]

    # 3) Draw the stacked bars for each pillar
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (xaxis, stratum_series) in enumerate(stratum_dict.items()):
        # remove zero-usage categories so we don't label them
        stratum_series = stratum_series[stratum_series > 0]
        
        names = stratum_series.index
        values = stratum_series.values
        
        bottom_accum = 0
        for name, val in zip(names, values):
            ax.bar(
                x=i,
                height=val,
                bottom=bottom_accum,
                width=0.4,
                color='white',
                edgecolor='black',
                linewidth=1
            )
            # Only label if >0
            label_y = bottom_accum + val / 2
            ax.text(
                i, label_y, str(name),
                ha='center', va='center', fontsize=8
            )
            bottom_accum += val

    # 4) Build color mapping if we have an alluvium
    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap_name)
    if alluvium:
        unique_vals = df[alluvium].unique()
        color_dict = {val: idx for idx, val in enumerate(unique_vals)}
    else:
        color_dict = {}

    # 5) Draw the flows
    if ignore_continuity:
        # Draw each pair of pillars separately
        for i in range(len(xaxis_names) - 1):
            left_col = xaxis_names[i]
            right_col = xaxis_names[i+1]
            agg_cols = [c for c in [alluvium, left_col, right_col] if c is not None]
            df_agg = df.groupby(agg_cols, observed=False, as_index=False)[y_name].sum()
            _plot_alluvium(df_agg, [left_col, right_col], y_name, alluvium, color_dict, cmap, ax, x_init=i)
    else:
        # Connect all pillars in one pass
        _plot_alluvium(df, xaxis_names, y_name, alluvium, color_dict, cmap, ax, x_init=0)

    ax.set_xticks(range(len(xaxis_names)))
    ax.set_xticklabels(xaxis_names)
    ax.set_xlim(-0.5, len(xaxis_names)-0.5)
    plt.tight_layout()
    return fig

###############################################################################
# (C) HELPER FOR DRAWING THE FLOWS
###############################################################################
def _plot_alluvium(df, xaxis_names, y_name, alluvium, color_dict, cmap, ax, x_init=0):
    """
    Internal helper that draws the sigmoid "flows" between consecutive pillars.
    We filter out zero usage if needed.
    """
    df = df.copy()
    
    # 1) Sort each pillar to build cumulative sums
    for xaxis in xaxis_names:
        df = df.sort_values(xaxis)
        # cumulative sum for each pillar
        df["y_" + xaxis] = df[y_name].cumsum().shift(1).fillna(0)
    
    # 2) Actually draw flows
    for i in range(len(xaxis_names) - 1):
        left_col = xaxis_names[i]
        right_col = xaxis_names[i+1]
        for _, row in df.iterrows():
            y_left = row["y_" + left_col]
            y_right = row["y_" + right_col]
            height = row[y_name]
            
            # skip 0 usage
            if height <= 0:
                continue
            
            color_key = row[alluvium] if alluvium else None
            
            xs, ys_under = calc_sigmoid_line(0.6, y_left, y_right)
            xs += i + 0.2
            ys_upper = ys_under + height
            
            # if color_key is absent from color_dict, use gray
            color = cmap(color_dict[color_key]) if (color_key in color_dict) else 'gray'
            ax.fill_between(xs + x_init, ys_under, ys_upper, color=color, alpha=0.7)
