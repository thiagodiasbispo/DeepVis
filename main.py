import os
from itertools import product

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Select,
    Slider,
    DataTable,
    TableColumn,
    HTMLTemplateFormatter,
    Range1d,
    Plot,
    ImageURL,
    NumericInput,
    SaveTool,
    ZoomOutTool,
    ZoomInTool,
    ResetTool,
)
from bokeh.palettes import d3
from bokeh.plotting import figure
from bokeh.util.sampledata import DataFrame
from cytoolz.functoolz import partial
from sklearn import cluster
from sklearn.manifold import TSNE

from common import *
from heatmap_util import plot_heatmap
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

all_options = tuple(product(testproblems, set(budgets) - {"oneshot"}, schedules))


def load_matrix_vector(file_name="desc_df.csv", calc_adj_matrix=False):
    if Path(file_name).exists():
        df_aux = pd.read_csv(file_name)
        with open("matriz_desc.npy", "rb") as f:
            return df_aux, np.load(f)

    matrixes = np.empty((len(all_options), len(optimizers) ** 2))

    print("Shape matrix: ", matrixes.shape)

    desc_list = [
        dict(problem=problem, budget=budget, schedule=sched)
        for (problem, budget, sched) in all_options
    ]

    desc_df = pd.DataFrame(desc_list)

    for i, (problem, budget, sched) in enumerate(all_options):
        df2 = get_optimized_vs_untuned_results(
            df,
            testproblem=problem,
            other_budget=budget,
            schedule=sched,
            metric_col=testproblems_loss_type[problem],
        )
        df2.fillna(0, inplace=True)

        if calc_adj_matrix:
            df2 = get_ordered_distance_matrix(df2)

        matrixes[i, :] = df2.to_numpy().reshape(1, -1)
        del df2

    with open("matriz_desc.npy", "wb") as f:
        np.save(f, matrixes)
    desc_df.to_csv(file_name, index=False)

    return desc_df, matrixes


desc_df, matrixes = load_matrix_vector(calc_adj_matrix=True)

matrix_tsne = TSNE(
    n_components=2, verbose=1, perplexity=60, n_iter=300, random_state=40
).fit_transform(matrixes)

print(matrix_tsne.shape)

desc_df["x"] = matrix_tsne[:, 0]
desc_df["y"] = matrix_tsne[:, 1]


labels_mapper = {
    "Budget": "budget",
    "Schedule": "schedule",
    "Test problem": "problem",
    "Detected groups": "auto",
}


class ScatterPlotSource(ColumnDataSource):
    @classmethod
    def from_df(cls, df: DataFrame, colors):
        source = ColumnDataSource()
        source.data = cls.data_from_df(df, colors)

    @staticmethod
    def data_from_df(df: DataFrame, colors: np.ndarray, legend_field_values: list):
        return dict(
            x=df["x"],
            y=df["y"],
            problem=df["problem"],
            budget=df["budget"],
            schedule=df["schedule"],
            color=colors,
            index=df.index,
            legend=legend_field_values,
        )

    @staticmethod
    def empty():
        return ColumnDataSource(data={})


def clustering(X, n_clusters):
    model = cluster.MiniBatchKMeans(n_clusters=n_clusters)
    model.fit(X)

    if hasattr(model, "labels_"):
        y_pred = model.labels_.astype(np.int)
    else:
        y_pred = model.predict(X)

    return X, y_pred


def uptade_option_to_color(attrname, old, new, source):
    color_by = labels_mapper[color_by_select.value]

    if color_by == "auto":
        clusters_slider.disabled = False
        n_clusters = int(clusters_slider.value)
        _, y_pred = clustering(matrix_tsne, n_clusters)
        options = list(y_pred)
        unique_options = list(set(y_pred))
        legend_field_values = [f"cluster{i:02}" for i in y_pred]
    else:
        legend_field_values = desc_df[color_by]
        clusters_slider.disabled = True
        clusters_slider.value = clusters_slider.start
        options = list(desc_df[color_by].values)
        unique_options = list(desc_df[color_by].unique())

    palette = d3["Category20"][len(unique_options)]

    colors = [palette[unique_options.index(i)] for i in options]

    source.data = ScatterPlotSource.data_from_df(
        desc_df, np.array(colors), legend_field_values
    )


def create_scatterplot(source):
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("problem", "@problem"),
        ("budget", "@budget"),
        ("schedule", "@schedule"),
    ]

    TOOLS = "wheel_zoom,box_select,reset,zoom_in,zoom_out,save,pan"

    # instantiating the figure object

    graph = figure(
        title="Matriz plot", tooltips=TOOLTIPS, tools=TOOLS, width=600, height=500
    )
    size = 10

    scatter = graph.scatter(
        source=source,
        size=size,
        x="x",
        y="y",
        fill_alpha=0.8,
        line_alpha=0.5,
        fill_color="color",
        legend_field="legend",
    )

    graph.legend.location = "bottom_left"

    return scatter, graph


scatter_source = ScatterPlotSource.empty()

color_options = list(labels_mapper.keys())
color_by_select = Select(
    value=color_options[0], title="Color by:", width=200, options=color_options
)

clusters_slider = Slider(
    title="Number of clusters", value=3.0, start=3.0, end=10.0, step=1, width=200
)

clusters_slider.disabled = True


first_index_input = NumericInput(
    title="First index", low=0, high=matrix_tsne.shape[0], width=200
)

second_index_input = NumericInput(
    title="Second index", low=0, high=matrix_tsne.shape[0], width=200
)


dt_source = ColumnDataSource(
    data=dict(budget=[""], problem=[""], color=[""], index=[""], schedule=[""])
)

images_source = ColumnDataSource(data=dict(url1=[""], url2=[""]))


def update_datatable(
    attr, old, new, dt_source, scatter_source,
):
    s = scatter_source.data
    data = dict(
        budget=s["budget"][new] if new else [""],
        schedule=s["schedule"][new] if new else [""],
        problem=s["problem"][new] if new else [""],
        color=s["color"][new] if new else [""],
        index=s["index"][new] if new else [""],
    )

    dt_source.data = data


def create_data_table(source):
    template = """                
                <div style="background-color:<%= color %>;text-align: center;"> 
                    <%= index %>  
                </div> 
             """

    color_formatter = HTMLTemplateFormatter(template=template)

    columns = [
        TableColumn(field="index", title="Index", formatter=color_formatter),
        TableColumn(field="problem", title="Problem"),
        TableColumn(field="budget", title="Budget"),
        TableColumn(field="schedule", title="Schedule"),
    ]

    data_table = DataTable(source=source, columns=columns, width=450, height=500)

    return data_table


def update_adj_matrixes(attr, old, new, source, img_source):
    try:
        first = int(first_index_input.value)
        second = int(second_index_input.value)
    except:
        return
    data = {}
    base_path = os.path.basename(os.path.dirname(__file__))
    for i, index in enumerate((first, second)):
        budget = source.data["budget"][index]
        problem = source.data["problem"][index]
        schedule = source.data["schedule"][index]

        fig_file = os.path.join("static", f"index{index:02}.png",)
        data[f"url{i+1}"] = [os.path.join(base_path, fig_file)]

        df_results = get_optimized_vs_untuned_results(
            df,
            testproblem=problem,
            other_budget=budget,
            schedule=schedule,
            metric_col=testproblems_loss_type[problem],
        )

        df_adj = get_ordered_distance_matrix(df_results)
        plot_heatmap(
            df_adj, figure_file=fig_file, create_colorbar=False,
        )

    img_source.data = data


def create_images_plot():
    xdr = Range1d(start=0, end=700)
    ydr = Range1d(start=0, end=700)
    plot1 = Plot(
        title=None,
        x_range=xdr,
        y_range=ydr,
        plot_width=500,
        plot_height=500,
        min_border=1,
        toolbar_location="below",
        tools=[SaveTool(), ZoomOutTool(), ZoomInTool(), ResetTool()],
    )

    plot2 = Plot(
        title=None,
        x_range=xdr,
        y_range=ydr,
        plot_width=500,
        plot_height=500,
        min_border=1,
        toolbar_location="below",
        tools=[SaveTool(), ZoomOutTool(), ZoomInTool(), ResetTool()],
    )

    image1 = ImageURL(w=1400, h=750, x=380, y=330, anchor="center_center", url="url1")
    plot1.add_glyph(images_source, image1)
    image2 = ImageURL(w=1400, h=750, x=380, y=330, anchor="center_center", url="url2")
    plot2.add_glyph(images_source, image2)

    return plot1, plot2


def build_gui():
    update_colors = partial(uptade_option_to_color, source=scatter_source)

    color_by_select.on_change("value", update_colors)
    clusters_slider.on_change("value", update_colors)

    update_adj_m = partial(
        update_adj_matrixes, source=scatter_source, img_source=images_source
    )

    first_index_input.on_change("value", update_adj_m)
    second_index_input.on_change("value", update_adj_m)

    controls = column(
        color_by_select, clusters_slider, first_index_input, second_index_input
    )

    scatter, graph = create_scatterplot(scatter_source)

    data_table = create_data_table(dt_source)

    update_table = partial(
        update_datatable, dt_source=dt_source, scatter_source=scatter_source
    )

    update_colors(None, None, None)
    scatter.data_source.selected.on_change("indices", update_table)

    img1, img2 = create_images_plot()
    images_row = row(img1, img2)
    display_grid = column(row(graph, data_table), images_row)
    return row(controls, display_grid)


root = build_gui()
curdoc().add_root(root)
curdoc().title = "Matrix plotting"
