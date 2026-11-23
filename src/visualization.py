"""Plotting functions for the exploratory analysis. Each one returns the figure and can save it."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TARGET_COLORS = {0: "#9aa5b1", 1: "#2a7ab0"}
TARGET_NAMES = {0: "neutral or dissatisfied", 1: "satisfied"}

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(fig, figures_dir, name):
    """Save a figure as PNG in the figures folder (created if needed)."""
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_target_balance(y_train, y_test, figures_dir=None):
    """Share of satisfied passengers in the train and test files."""
    shares = pd.DataFrame({
        "train": y_train.map(TARGET_NAMES).value_counts(normalize=True),
        "test": y_test.map(TARGET_NAMES).value_counts(normalize=True),
    }).T
    fig, ax = plt.subplots(figsize=(6, 3))
    shares.plot.barh(stacked=True, ax=ax, color=[TARGET_COLORS[0], TARGET_COLORS[1]], width=0.6)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.1%}" for v in container.datavalues], label_type="center", color="white")
    ax.set(xlabel="share of passengers", xlim=(0, 1), title="Target balance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    save_figure(fig, figures_dir, "01_target_balance")
    return fig


def plot_rate_by_category(X, y, columns, figures_dir=None):
    """Satisfaction rate for each level of each categorical column, with the level counts."""
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.6), sharey=True)
    overall = y.mean()
    for ax, column in zip(axes, columns):
        stats = y.groupby(X[column]).agg(["mean", "size"]).sort_values("mean")
        ax.bar(stats.index.astype(str), stats["mean"], color=TARGET_COLORS[1])
        ax.axhline(overall, color="black", linestyle="--", linewidth=1)
        for i, (rate, count) in enumerate(zip(stats["mean"], stats["size"])):
            ax.text(i, rate + 0.02, f"{rate:.0%}\n(n={count:,})", ha="center", fontsize=8)
        ax.set(title=column, ylim=(0, 1), xlabel="")
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("satisfaction rate")
    fig.suptitle(f"Satisfaction rate by passenger and trip category (dashed line = overall {overall:.0%})")
    fig.tight_layout()
    save_figure(fig, figures_dir, "02_rate_by_category")
    return fig


def plot_numeric_by_target(X, y, columns, log_columns=(), figures_dir=None):
    """Distribution of each numeric column for satisfied vs other passengers."""
    fig, axes = plt.subplots(1, len(columns), figsize=(4.5 * len(columns), 3.6))
    for ax, column in zip(axes, columns):
        values = np.log1p(X[column]) if column in log_columns else X[column]
        for label in (0, 1):
            sns.histplot(values[y == label], ax=ax, bins=40, stat="density", element="step",
                         color=TARGET_COLORS[label], label=TARGET_NAMES[label], fill=True, alpha=0.3)
        ax.set(title=f"log(1 + {column})" if column in log_columns else column, xlabel="")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir, "03_numeric_by_target")
    return fig


def plot_rate_by_bins(X, y, column, bins, figures_dir=None, name=None):
    """Satisfaction rate per interval of a numeric column."""
    groups = pd.cut(X[column], bins=bins)
    stats = y.groupby(groups, observed=True).agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(stats.index.astype(str), stats["mean"], color=TARGET_COLORS[1])
    for i, (rate, count) in enumerate(zip(stats["mean"], stats["size"])):
        ax.text(i, rate + 0.02, f"{rate:.0%}\n(n={count:,})", ha="center", fontsize=8)
    ax.set(ylim=(0, 1), ylabel="satisfaction rate", xlabel=column, title=f"Satisfaction rate by {column}")
    save_figure(fig, figures_dir, name or f"rate_by_{column}")
    return fig


def plot_rating_heatmap(X, y, ratings, figures_dir=None):
    """Satisfaction rate for each rating level (0 is treated as not applicable) of each service."""
    order = X[ratings].corrwith(y).sort_values(ascending=False).index
    rates = pd.DataFrame({column: y.groupby(X[column]).mean() for column in order}).T
    counts = pd.DataFrame({column: X[column].value_counts() for column in order}).T
    # Levels seen fewer than 30 times give unreliable rates: they are hidden.
    rates = rates.where(counts >= 30)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(rates, annot=True, fmt=".0%", cmap="RdBu", center=y.mean(), vmin=0, vmax=1,
                cbar_kws={"label": "satisfaction rate"}, ax=ax)
    ax.set(xlabel="rating (0 = treated as not applicable)", ylabel="",
           title="Satisfaction rate by service rating\n(sorted by correlation with the target)")
    ax.grid(False)
    save_figure(fig, figures_dir, "05_rating_heatmap")
    return fig


def plot_feature_relevance(relevance, figures_dir=None, name="06_feature_relevance"):
    """Horizontal bar charts of several relevance measures (one column of `relevance` per chart)."""
    fig, axes = plt.subplots(1, relevance.shape[1], figsize=(5 * relevance.shape[1], 0.32 * len(relevance) + 1.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    ordered = relevance.sort_values(relevance.columns[0])
    for ax, measure in zip(axes, ordered.columns):
        colors = [TARGET_COLORS[1] if v >= 0 else "#c0504d" for v in ordered[measure]]
        ax.barh(ordered.index, ordered[measure], color=colors)
        ax.set(title=measure, xlabel="")
    fig.tight_layout()
    save_figure(fig, figures_dir, name)
    return fig


def plot_correlation_heatmap(frame, figures_dir=None, name="07_rating_correlations", title=None):
    """Clustered Spearman correlation matrix (rank correlation suits 1-5 ratings)."""
    corr = frame.corr(method="spearman")
    grid = sns.clustermap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                          annot_kws={"size": 7}, figsize=(9, 9), cbar_pos=(0.02, 0.83, 0.03, 0.15))
    grid.fig.suptitle(title or "Spearman correlation between service ratings (clustered)", y=1.02)
    save_figure(grid.fig, figures_dir, name)
    return grid.fig


def plot_interaction(X, y, row, column, figures_dir=None, name=None):
    """Satisfaction rate for each combination of two categorical columns."""
    rates = y.groupby([X[row], X[column]]).mean().unstack()
    counts = y.groupby([X[row], X[column]]).size().unstack()
    labels = rates.map(lambda v: f"{v:.0%}") + counts.map(lambda n: f"\n(n={n:,})")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sns.heatmap(rates, annot=labels, fmt="", cmap="RdBu", center=y.mean(), vmin=0, vmax=1,
                cbar_kws={"label": "satisfaction rate"}, ax=ax)
    ax.set(title=f"Satisfaction rate: {row} x {column}")
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    save_figure(fig, figures_dir, name or f"interaction_{row}_{column}")
    return fig


def plot_delay_relationship(X, figures_dir=None):
    """Arrival vs departure delay, to justify filling one with the other."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(X["departure_delay_in_minutes"], X["arrival_delay_in_minutes"], s=2, alpha=0.15, color=TARGET_COLORS[1])
    limit = X[["departure_delay_in_minutes", "arrival_delay_in_minutes"]].max().max()
    ax.plot([0, limit], [0, limit], color="black", linewidth=1, linestyle="--", label="arrival = departure")
    ax.set(xlabel="departure delay (min)", ylabel="arrival delay (min)", title="Arrival vs departure delay")
    ax.legend()
    save_figure(fig, figures_dir, "08_delay_relationship")
    return fig
