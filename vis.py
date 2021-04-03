import scipy.cluster.hierarchy as sch


def get_ordered_distance_matrix(df):
    m = df.to_numpy()
    Y = sch.linkage(m, method="centroid")
    Z = sch.dendrogram(Y, orientation="right", no_plot=True)

    index = Z["leaves"]

    df = df.iloc[index, :]
    df = df.iloc[:, index]

    return df
