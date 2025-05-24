import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # K-Means Clustering Animation & Customer Segmentation

    ## Overview

    This [Marimo](https://marimo.io) notebook demonstrates the **k-means clustering algorithm** using an interactive, animated Plotly visualisation. The notebook is fully reactive, leveraging Marimo’s capabilities to provide a seamless, modern user experience. 

    **Features:**  

    - Synthetic data points are generated in two dimensions.
    - The k-means algorithm is run step-by-step, updating cluster assignments and centroids at each iteration.
    - The animation shows how clusters evolve, with each frame representing a new iteration.
    - Users can play, pause, and loop the animation using intuitive controls.

    ## What is K-Means Clustering?

    **K-means** is an unsupervised machine learning algorithm that partitions data into $k$ clusters, each represented by a centroid. The algorithm iteratively:

    1. Assigns each point to the nearest centroid.
    2. Updates centroids as the mean of assigned points.
    3. Repeats until assignments stabilise or a set number of iterations is reached.

    In the animation:

    - **Points** are coloured by their current cluster assignment.
    - **Centroids** are shown as large 'X' markers, updating with each iteration.

    In clustering algorithms, like k-means, the **centroid** of a cluster is the average position of all the data points in that group, acting as the “centre” of the cluster.

    ## K-Means and Customer Segmentation

    **Customer segmentation** is a key application of k-means in business analytics. It involves grouping customers into distinct segments based on shared characteristics (such as age, spending, or product preference).

    **Why k-means?** It automatically finds natural groupings in multi-dimensional customer data, enabling targeted marketing and personalised service.

    ### Mapping Multi-Attribute Customers to 2D

    - In practice, customers have many attributes (e.g., age, income, behaviour scores).
    - K-means works natively in any number of dimensions.
    - For visualisation, high-dimensional data are often projected into 2D or 3D (using methods like PCA or t-SNE, or by selecting two key features) to illustrate clustering results.
    - In this notebook, we use synthetic 2D data for clarity, but the algorithm and code generalise to higher dimensions (lots of different customer characteristics).

    ### How K-Means Works in Multi-Dimensions

    - **Input:** Each customer is a vector in $n$-dimensional space (where $n$ is the number of attributes).
    - **Process:** K-means computes distances and centroids in this multi-dimensional space, not just in 2D.
    - **Output:** Each customer is assigned to the nearest cluster, regardless of how many attributes are used.

    ## About Marimo & Plotly Integration

    - **Marimo** is a reactive Python notebook platform supporting modern plotting libraries, including Plotly, for interactive data exploration.
    - **Plotly** provides high-quality, interactive visualisations and smooth animations, ideal for illustrating iterative algorithms like k-means.
    - The notebook uses Plotly’s animation features and Marimo’s reactivity, allowing you to interact with the visualisation and extend it to select and analyse clusters in real time.

    **To run the notebook:**  

    - Ensure you have installed `marimo` and `plotly`.
    - Launch the notebook and use the controls to explore k-means clustering in action.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    import plotly.graph_objects as go
    return KMeans, go, make_blobs, mo


@app.function
def get_colour_palette(n_clusters):
    """Return a list of visually distinct colours (hex codes)."""
    base_colours = [
        '#636EFA',  # blue
        '#EF553B',  # red
        '#00CC96',  # green
        '#AB63FA',  # purple
        '#FFA15A',  # orange
        '#19D3F3',  # cyan
        '#FF6692'   # pink
    ]
    # Repeat if more clusters than base colours
    return (base_colours * ((n_clusters // len(base_colours)) + 1))[:n_clusters]


@app.cell
def _(go):
    def create_kmeans_animation(X, labels_history, centers_history, colors, n_iter, n_clusters, std_dev):
        """
        Build Plotly animation frames for k-means clustering, with parameter annotations.
        """
        frames = []
        for i in range(n_iter):
            scatter = go.Scatter(
                x=X[:, 0], y=X[:, 1],
                mode='markers',
                marker=dict(
                    color=[colors[l] for l in labels_history[i]],
                    size=6,
                    line=dict(width=1, color='black')
                ),
                showlegend=False
            )
            centroids = go.Scatter(
                x=centers_history[i][:, 0], y=centers_history[i][:, 1],
                mode='markers',
                marker=dict(color=colors, size=25, symbol='x'),
                showlegend=False
            )
            annotation_text = (
                f"Clusters (k): {n_clusters} &nbsp;|&nbsp; "
                f"Std Dev: {std_dev} &nbsp;|&nbsp; "
                f"Frame: {i+1} / {n_iter}"
            )
            frames.append(go.Frame(
                data=[scatter, centroids],
                name=f"iter_{i+1}",
                layout=go.Layout(
                    title_text=f"K-means Iteration {i+1}",
                    annotations=[
                        dict(
                            text=annotation_text,
                            xref="paper", yref="paper",
                            x=0.5, y=1.08, showarrow=False,
                            font=dict(size=14)
                        )
                    ]
                )
            ))
        return frames

    return (create_kmeans_animation,)


@app.cell
def _(go):
    def build_figure(X, centers_history, colours, frames):
        """Create the initial Plotly figure with animation controls."""
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=X[:, 0], y=X[:, 1],
                    mode='markers',
                    marker=dict(color='grey', size=6, line=dict(width=1, color='black')),
                    showlegend=False
                ),
                go.Scatter(
                    x=centers_history[0][:, 0], y=centers_history[0][:, 1],
                    mode='markers',
                    marker=dict(color=colours, size=25, symbol='x'),
                    showlegend=False
                ),
            ],
            layout=go.Layout(
                title="K-means Iteration 1",
                xaxis=dict(range=[X[:,0].min()-1, X[:,0].max()+1]),
                yaxis=dict(range=[X[:,1].min()-2, X[:,1].max()+2]),
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True, "loop": True}],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }
                ]
            ),
            frames=frames
        )
        return fig
    return (build_figure,)


@app.cell
def _(make_blobs):
    def generate_data(n_samples, n_clusters, std_dev, random_state=42):
        """Generate synthetic data for clustering."""
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            cluster_std=std_dev,
            random_state=random_state
        )
        return X
    return (generate_data,)


@app.cell
def _(KMeans):
    def run_kmeans_iterations(X, n_clusters, n_iter):
        """Run k-means step by step, collecting labels and centroids for each iteration."""
        labels_history = []
        centers_history = []
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=1, random_state=42)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        for _ in range(n_iter):
            kmeans = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=1, random_state=42)
            kmeans.fit(X)
            labels_history.append(kmeans.labels_.copy())
            centers_history.append(kmeans.cluster_centers_.copy())
            centers = kmeans.cluster_centers_
        return labels_history, centers_history
    return (run_kmeans_iterations,)


@app.cell
def _(mo):
    # Interactive sliders for user control
    n_clusters_slider = mo.ui.slider(2, 7, 1, value=5, label="Number of clusters (k)")
    n_samples_slider = mo.ui.slider(50, 500, 50, value=300, label="Number of samples")
    std_dev_slider = mo.ui.slider(0.5, 3.0, 0.1, value=1.8, label="Cluster standard deviation")
    #n_iter_slider = mo.ui.slider(3, 15, 1, value=8, label="K-means iterations (animation frames)")
    n_iter = 20
    return n_clusters_slider, n_iter, n_samples_slider, std_dev_slider


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Instructions

    Use the sliders to adjust the number of clusters, samples, standard deviation, and animation length. 

    The animation below shows how k-means iteratively finds clusters in 2D data.
    """
    )
    return


@app.cell
def _(mo, n_clusters_slider, n_samples_slider, std_dev_slider):
    # Show the slider controls
    mo.hstack([
        n_clusters_slider,
        n_samples_slider,
        std_dev_slider,
        # n_iter_slider
    ])
    return


@app.cell
def _(
    build_figure,
    create_kmeans_animation,
    generate_data,
    n_clusters_slider,
    n_iter,
    n_samples_slider,
    run_kmeans_iterations,
    std_dev_slider,
):
    # Generate data based on user input
    X = generate_data(
        n_samples=n_samples_slider.value,
        n_clusters=n_clusters_slider.value,
        std_dev=std_dev_slider.value
    )

    # Run k-means and collect history
    labels_history, centers_history = run_kmeans_iterations(
        X, n_clusters_slider.value, n_iter # _slider.value
    )

    # Get colours for clusters
    colours = get_colour_palette(n_clusters_slider.value)

    # Build animation frames
    frames = create_kmeans_animation(
        X, labels_history, centers_history, colours, n_iter, n_clusters_slider.value, std_dev_slider.value
    )

    # Build and show figure
    fig = build_figure(X, centers_history, colours, frames)
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
