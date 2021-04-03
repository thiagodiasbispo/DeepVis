import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def heatmap(
    data,
    row_labels,
    col_labels,
    xlabel=None,
    ylabel=None,
    ax=None,
    xy_labelsize=(20, 20),
    row_colum_fontsize=(20, 20),
    cbar_kw={},
    cbarlabel="",
    **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    create_colorbar = kwargs.pop("create_colorbar", True)

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    if create_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=-30,
        ha="right",
        fontsize=row_colum_fontsize[0],
        rotation_mode="anchor",
    )

    plt.setp(ax.get_yticklabels(), fontsize=row_colum_fontsize[1])

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel(xlabel, fontsize=xy_labelsize[0])
    ax.set_ylabel(ylabel, fontsize=xy_labelsize[1])

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heatmap(
    df,
    threshold=20.6,
    size_inches=(20, 10),
    fig=None,
    figure_file=None,
    number_fontsize=10,
    ax=None,
    **kwargs
):

    if not ax:
        fig, ax = plt.subplots()

    fig = ax.figure

    # im = ax.imshow(harvest)number_fontsize
    columns = df.columns
    matrix = df.to_numpy()

    # norm = mpl.colors.Normalize(vmin=0, vmax=matrix.max())
    norm = mpl.colors.TwoSlopeNorm(vmin=matrix.min(), vmax=matrix.max(), vcenter=0)

    row_labels = kwargs.pop("row_labels", columns)
    col_labels = kwargs.pop("col_labels", columns)

    im, cbar = heatmap(
        matrix,
        row_labels,
        col_labels,
        ax=ax,
        norm=norm,
        cmap="coolwarm",
        cbarlabel="Performance Improvment",
        **kwargs
    )

    texts = annotate_heatmap(
        im, valfmt="{x:.1f}", threshold=threshold, fontsize=number_fontsize
    )

    fig.set_size_inches(*size_inches)

    if figure_file:
        plt.savefig(figure_file)


def plot_baseline_x_improoved(
    baseline, ours, other_budget, base_budget="oneshot", figure_name=None
):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    row_colum_fontsize = (30, 30)
    plot_heatmap(
        baseline,
        xlabel=other_budget,
        row_colum_fontsize=row_colum_fontsize,
        ylabel=base_budget,
        number_fontsize=35,
        ax=ax1,
        create_colorbar=False,
    )
    plot_heatmap(
        ours,
        xlabel=other_budget,
        number_fontsize=35,
        ylabel=base_budget,
        row_colum_fontsize=row_colum_fontsize,
        # row_labels=[],
        ax=ax2,
        create_colorbar=False,
    )

    fig.set_size_inches(80, 60)

    plt.show()
