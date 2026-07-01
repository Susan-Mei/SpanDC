import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

sns.set_theme()
sns.set_color_codes()


# Plots a time series with annotations
def plot_time_series(title, time_series, change_points=None, labels=None, file_path=None, font_size=18):
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)

    if change_points is None:
        change_points = np.zeros(0, dtype=int)

    if labels is None:
        labels = np.zeros(change_points.shape[0] + 1, dtype=int)

    plt.clf()
    fig, axes = plt.subplots(
        time_series.shape[1],
        sharex=True,
        gridspec_kw={'hspace': .15},
        figsize=(20, time_series.shape[1] * 2)
    )

    if time_series.shape[1] == 1:
        axes = [axes]

    label_colours = {}
    idx = 0

    for activity in labels:
        if activity not in label_colours:
            label_colours[activity] = f"C{idx}"
            idx += 1

    for dim, ax in enumerate(axes):
        ts = time_series[:, dim]

        if len(ts) > 0:
            segments = [0] + change_points.tolist() + [ts.shape[0]]
            for idx in np.arange(0, len(segments) - 1):
                ax.plot(
                    np.arange(segments[idx], segments[idx + 1]),
                    ts[segments[idx]:segments[idx + 1]],
                    c=label_colours[labels[idx]]
                )

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

    axes[0].set_title(title, fontsize=font_size)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax


# Plots a time series with annotated state sequences
def plot_state_detection(title, time_series, state_seq, change_points=None, labels=None, ylabels=None, file_path=None,
                         font_size=18):
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)

    if state_seq.ndim == 1:
        state_seq = state_seq.reshape(-1, 1)

    if change_points is None:
        change_points = np.zeros(0, dtype=int)

    if labels is None:
        labels = np.zeros(change_points.shape[0] + 1, dtype=int)

    plt.clf()
    fig, axes = plt.subplots(
        time_series.shape[1] + state_seq.shape[1],
        sharex=True,
        gridspec_kw={'hspace': .5},
        figsize=(10, time_series.shape[1] + state_seq.shape[1])
    )

    label_colours = {}
    state_colours = {}

    idx = 0

    for label in labels:
        if label not in label_colours:
            label_colours[label] = f"C{idx}"
            idx += 1

    for label in np.unique(state_seq):
        state_colours[label] = f"C{idx}"
        idx += 1

    for dim, ax in enumerate(axes):
        if dim < time_series.shape[1]:
            series = time_series[:, dim]
        else:
            series = state_seq[:, dim - time_series.shape[1]]

        segments = [0] + change_points.tolist() + [series.shape[0]]
        colors = label_colours
        annotation = labels

        if len(series) > 0:
            for idx in np.arange(0, len(segments) - 1):
                ax.plot(
                    np.arange(segments[idx], segments[idx + 1]),
                    series[segments[idx]:segments[idx + 1]],
                    c=colors[annotation[idx]]
                )

        if ylabels is not None and dim < len(ylabels):
            ax.set_ylabel(ylabels[dim], fontsize=font_size)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        ax.locator_params(axis='y', nbins=min(5, np.unique(series).shape[0]))
        ax.tick_params(axis='y', labelsize=font_size // 2)

        ax.grid(True)

    axes[0].set_title(title, fontsize=font_size)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax


# Plots transition graph from state sequence
def plot_state_transition_graph(title, state_seq, file_path=None, font_size=14):
    unique_states, state_indices = np.unique(state_seq, return_inverse=True)
    n_states = len(unique_states)

    transition_counts = np.zeros((n_states, n_states))
    for (i, j) in zip(state_indices[:-1], state_indices[1:]):
        transition_counts[i, j] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transition_counts, row_sums, out=np.zeros_like(transition_counts), where=row_sums != 0)

    G = nx.DiGraph()
    for i, state_from in enumerate(unique_states):
        G.add_node(state_from)
        for j, state_to in enumerate(unique_states):
            prob = transition_probs[i, j]
            if prob > 0:
                G.add_edge(state_from, state_to, weight=prob)

    node_colors = []

    for node in G.nodes():
        if node == state_seq[0]:
            node_colors.append("lightgreen")
        elif node == state_seq[-1]:
            node_colors.append("lightblue")
        else:
            node_colors.append("lightgrey")

    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    _, ax = plt.subplots(figsize=(7, 7))
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=1000, node_color=node_colors, arrowsize=20, font_size=12)

    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color="black", font_size=12)

    ax.set_title(title, fontsize=font_size)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")

    return ax
