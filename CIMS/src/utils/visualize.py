import numpy as np
import pandas as pd
import umap.umap_ as umap
import plotly.express as px
import numbers
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import torch
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from matplotlib import gridspec

def compute_umap_embedding(
    results_df,
    subset=20000,
    n_neighbors=30,
    min_dist=0.01,
    spread=2.0,
    random_state=None,
    dim=2,
):
    """
    Subsamples the dataframe and computes UMAP embedding.

    Returns:
        subset_df:  the filtered dataframe
        embedding:  numpy array (N, dim)
    """

    # Subsampling
    if subset < len(results_df):
        subset_df = results_df.sample(subset, replace=False)
    else:
        subset_df = results_df

    # Extract feature matrix
    X = np.array(subset_df.features.tolist())
    if X.ndim == 4:
        X = X.mean(axis=(2,3))
        print(X.shape)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=dim,
        random_state=random_state,
        metric="euclidean"
    )

    embedding = reducer.fit_transform(X)
    
    subset_df['UMAP_X'] = embedding[:, 0]
    subset_df['UMAP_Y'] = embedding[:, 1]
    if dim == 3:
        subset_df['UMAP_Z'] = embedding[:, 2]

    return subset_df


def plot_umap_embedding(
    results_df,
    display_col,
    cmap=None,
    dim=2,
    target=None,
    s=1,
    norm=100,
    alpha=0.5,
    save_dir=None
):
    display = results_df[display_col].values if display_col is not None else np.array([1]*len(results_df))
    is_numeric = np.issubdtype(display.dtype, np.number)

    if is_numeric:
        display = np.clip(display / np.percentile(display, norm), 0, 1)

    # Optional filtering by target
    if target is not None:
        display = display.copy()
        display[display != target] = "other"
        if isinstance(cmap, dict):
            cmap["other"] = "#AAAAAA"

    df_plot = pd.DataFrame({
        "UMAP_X": results_df["UMAP_X"],
        "UMAP_Y": results_df["UMAP_Y"],
        "color": display,
    })
    if dim == 3:
        df_plot["UMAP_Z"] = results_df["UMAP_Z"]

    if target is None:
        plot_title = f"UMAP colored by: {display_col}"
    else:
        plot_title = f"UMAP colored by: {display_col} (target={target})"

    # 2D
    if dim == 2:
        if is_numeric:
            fig = px.scatter(
                df_plot,
                x="UMAP_X",
                y="UMAP_Y",
                color="color",
                color_continuous_scale="Viridis",
                opacity=alpha,  # <--- use alpha here
                width=800,
                height=700,
                title=plot_title,
            )
        else:
            if isinstance(cmap, dict):
                fig = px.scatter(
                    df_plot,
                    x="UMAP_X",
                    y="UMAP_Y",
                    color="color",
                    color_discrete_map=cmap,
                    opacity=alpha,  # <---
                    width=800,
                    height=700,
                    title=plot_title,
                )
            else:
                fig = px.scatter(
                    df_plot,
                    x="UMAP_X",
                    y="UMAP_Y",
                    color="color",
                    opacity=alpha,  # <---
                    width=800,
                    height=700,
                    title=plot_title,
                )

        fig.update_traces(marker=dict(size=s, opacity=alpha))  # <--- not alpha

    # 3D
    else:
        if is_numeric:
            fig = px.scatter_3d(
                df_plot,
                x="UMAP_X",
                y="UMAP_Y",
                z="UMAP_Z",
                color="color",
                color_continuous_scale="Viridis",
                opacity=alpha,  # <---
                width=900,
                height=800,
                title=plot_title,
            )
        else:
            if isinstance(cmap, dict):
                fig = px.scatter_3d(
                    df_plot,
                    x="UMAP_X",
                    y="UMAP_Y",
                    z="UMAP_Z",
                    color="color",
                    color_discrete_map=cmap,
                    opacity=alpha,  # <---
                    width=900,
                    height=800,
                    title=plot_title,
                )
            else:
                fig = px.scatter_3d(
                    df_plot,
                    x="UMAP_X",
                    y="UMAP_Y",
                    z="UMAP_Z",
                    color="color",
                    opacity=alpha,  # <---
                    width=900,
                    height=800,
                    title=plot_title,
                )

        fig.update_traces(marker=dict(size=s, opacity=alpha))

    fig.show()
    
    if save_dir is not None:
        fig.write_html(f"{save_dir}/UMAP_{display_col}_{dim}D.html")
    
    return fig