from itertools import product
from math import pi

import pandas as pd
from bokeh.io import curdoc
from bokeh.models import (
    BasicTicker,
    ColorBar,
    LinearColorMapper,
    PrintfTickFormatter,
)
from bokeh.palettes import RdYlBu
from bokeh.plotting import figure

from common import *
from report import get_optimized_vs_untuned_results
from vis import get_ordered_distance_matrix


def load_results_new_paper_version():
    df = pd.read_csv(
        "results_new_paper_version.csv", sep=";", usecols=metrics_cols + params_cols
    )
    df.optimizer = df.optimizer.apply(lambda x: optimizers_to_small_name_dict[x])

    print("Dataset size: ", len(df))

    return df


df = load_results_new_paper_version()

df_compare = get_optimized_vs_untuned_results(
    df,
    testproblem="cifar10_3c3d",
    other_budget="medium_budget",
    schedule="none",
    metric_col=testproblems_loss_type["cifar10_3c3d"],
)


mall_options = tuple(product(testproblems, set(budgets) - {"oneshot"}, schedules))

df_adj = get_ordered_distance_matrix(df_compare)

adj_data = [
    (opt1, opt2, row[opt2]) for (opt1, row) in df_adj.iterrows() for opt2 in row.keys()
]

adj_data = pd.DataFrame(adj_data, columns=["optimizer1", "optimizer2", "improvment"])


opt_x = list(adj_data.optimizer1)
opt_y = list(adj_data.optimizer1)


colors = ["blue", "gray", "orange"]

mapper = LinearColorMapper(
    palette=RdYlBu[9], low=df_adj.min().min(), high=df_adj.max().max()
)

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p = figure(
    title="Optimizers comparation",
    x_range=list(df_adj.columns),
    y_range=list(df_adj.index),
    x_axis_location="above",
    plot_width=500,
    plot_height=500,
    tools=TOOLS,
    toolbar_location="below",
    tooltips=[
        ("Optimizers", "@optimizer1 | @optimizer2"),
        ("Improvment", "@improvment"),
    ],
)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "7px"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(
    x="optimizer1",
    y="optimizer2",
    width=1,
    height=1,
    source=adj_data,
    fill_color={"transform": mapper, "field": "improvment"},
    line_color=None,
)

color_bar = ColorBar(
    color_mapper=mapper,
    major_label_text_font_size="7px",
    ticker=BasicTicker(desired_num_ticks=len(colors)),
    formatter=PrintfTickFormatter(format="%d"),
    label_standoff=6,
    border_line_color=None,
)
p.add_layout(color_bar, "right")

curdoc().add_root(p)
curdoc().title = "Heatmap"
