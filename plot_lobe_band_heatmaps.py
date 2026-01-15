import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def plot_lobe_band_heatmaps(abs_matrix, pct_matrix, lobes, bands):
    """
    Create heatmaps & return the figure object.
    Saving happens in MAIN.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap_choice = "Greens"

    # ---------- ABSOLUTE POWER TOTALS ----------
    row_sum_abs = np.sum(abs_matrix, axis=1)
    col_sum_abs = np.sum(abs_matrix, axis=0)

    extended_abs = np.zeros((abs_matrix.shape[0] + 1,
                             abs_matrix.shape[1] + 1))

    extended_abs[:-1, :-1] = abs_matrix
    extended_abs[:-1, -1] = row_sum_abs
    extended_abs[-1, :-1] = col_sum_abs
    extended_abs[-1, -1] = np.sum(col_sum_abs)

    if isinstance(bands, dict):
        bands = list(bands.keys())

    row_labels_abs = lobes + ["Total"]
    col_labels_abs = bands + ["Total"]

    # ---------- PERCENTAGE TOTALS + LOBE-WISE TBR ----------
    row_sum_pct = np.sum(pct_matrix, axis=1)
    col_sum_pct = np.sum(pct_matrix, axis=0)

    # Indices of theta and beta in bands list
    theta_idx = bands.index("theta")
    beta_idx  = bands.index("beta")

    # Lobe-wise TBR (using % matrix, equivalent to abs theta/beta)
    TBR_lobes = pct_matrix[:, theta_idx] / (pct_matrix[:, beta_idx] + 1e-12)
    TBR_mean  = np.mean(TBR_lobes)

    # extended_pct shape = (lobes+1) × (bands + TBR + Total%)
    extended_pct = np.zeros((pct_matrix.shape[0] + 1,
                             pct_matrix.shape[1] + 2))

    # Core % matrix
    extended_pct[:-1, :-2] = pct_matrix

    # Row totals (next-to-last column)
    extended_pct[:-1, -2] = row_sum_pct

    # Column totals (last row, band columns only)
    extended_pct[-1, :-2] = col_sum_pct

    # Bottom-right of "Total %" column = 100
    extended_pct[-1, -2] = 100.0

    # TBR column (last column)
    extended_pct[:-1, -1] = TBR_lobes
    extended_pct[-1, -1]  = TBR_mean

    row_labels_pct = lobes + ["Total"]
    col_labels_pct = bands + ["TBR","Total%" ]

    # ---- Absolute Power ----
    # ----- build mask -----
    mask_abs = np.zeros_like(extended_abs, dtype=bool)
    mask_abs[-1, :] = True  # mask last row
    mask_abs[:, -1] = True  # mask last column

    ax_abs = sns.heatmap(
        extended_abs,
        mask=mask_abs,
        ax=axes[0],
        cmap=cmap_choice,
        vmin=0,
        annot=False,  # we'll add labels ourselves
        cbar=True,
        fmt=".2f",
        xticklabels=col_labels_abs,
        yticklabels=row_labels_abs
    )
    axes[0].set_title("Absolute Band Power (Lobe × Band)")

    # ---- ABS core annotations ----
    for i in range(len(lobes)):
        for j in range(len(bands)):
            ax_abs.text(j + 0.5, i + 0.5,
                        f"{abs_matrix[i, j]:.2f}",
                        ha='center', va='center')

    # ---- Relative Power ----
    # ----- build mask -----

    mask_pct = np.zeros_like(extended_pct, dtype=bool)
    #mask_pct[-1, :] = True
    mask_pct[:, -2] = True

    ax_pct = sns.heatmap(
        extended_pct,
        mask=mask_pct,
        ax=axes[1],
        cmap=cmap_choice,
        vmin=0,
        vmax=35,
        annot=False,
        cbar=True,
        fmt=".2f",
        xticklabels=col_labels_pct,
        yticklabels=row_labels_pct
    )
    axes[1].set_title("Relative Band Power % (Lobe × Band)")

    # ---- Annotate band % ----
    for i in range(len(lobes)):
        for j in range(len(bands)):
            ax_pct.text(j + 0.5, i + 0.5,
                        f"{pct_matrix[i, j]:.2f}",
                        ha='center', va='center')

    # ---- Annotate TBR ----
    for i in range(len(lobes)):
        ax_pct.text(len(bands) + 0.5, i + 0.5,
                    f"{TBR_lobes[i]:.2f}",
                    ha='center', va='center')

    # ---- Annotate row totals ----
    for i in range(len(lobes)):
        ax_pct.text(len(bands) + 1.5, i + 0.5,
                    f"{row_sum_pct[i]:.2f}",
                    ha='center', va='center')

    # ---- Annotate last row ----
    for j in range(len(bands)):
        ax_pct.text(j + 0.5, len(lobes) + 0.5,
                    f"{col_sum_pct[j]:.2f}",
                    ha='center', va='center')

    ax_pct.text(len(bands) + 0.5, len(lobes) + 0.5,
                f"{TBR_mean:.2f}",
                ha='center', va='center')

    ax_pct.text(len(bands) + 1.5, len(lobes) + 0.5,
                "100.00",
                ha='center', va='center')

    # ---- Add row totals ----
    for i in range(len(lobes)):
        ax_abs.text(len(bands) + 0.5, i + 0.5,
                    f"{row_sum_abs[i]:.2f}",
                    ha='center', va='center')

        ax_pct.text(len(bands) + 1.5, i + 0.5, #1.5 as we have extra TBR col
                    f"{row_sum_pct[i]:.2f}",
                    ha='center', va='center')

    # ---- Add column totals ----
    for j in range(len(bands)):
        ax_abs.text(j + 0.5, len(lobes) + 0.5,
                    f"{col_sum_abs[j]:.2f}",
                    ha='center', va='center')

        ax_pct.text(j + 0.5, len(lobes) + 0.5,
                    f"{col_sum_pct[j]:.2f}",
                    ha='center', va='center')

    # ---- Add grand totals ----
    ax_abs.text(len(bands) + 0.5, len(lobes) + 0.5,
                f"{np.sum(col_sum_abs):.2f}",
                ha='center', va='center')

    # ---- Add TBR mean bottom cell ----
    ax_pct.text(len(bands) + 0.5, len(lobes) + 0.5,
                f"{TBR_mean:.2f}",
                ha='center', va='center')

    # column total block
    ax_abs.add_patch(
        patches.Rectangle(
            (len(bands), 0),
            1, len(lobes),
            linewidth=1,
            edgecolor='black',
            facecolor='whitesmoke'
        )
    )

    # row total block
    ax_abs.add_patch(
        patches.Rectangle(
            (0, len(lobes)),
            len(bands), 1,
            linewidth=1,
            edgecolor='black',
            facecolor='whitesmoke'
        )
    )

    # bottom right corner
    ax_abs.add_patch(
        patches.Rectangle(
            (len(bands), len(lobes)),
            1, 1,
            linewidth=2,
            edgecolor='black',
            facecolor='gainsboro'
        )
    )

    ax_pct.add_patch(
        patches.Rectangle(
            (len(bands), 0), # 5,0
            1, len(lobes),
            linewidth=2,
            edgecolor='black',
            facecolor='whitesmoke'
        )
    )

    ax_pct.add_patch(
        patches.Rectangle(
            (len(bands)+1, 0), # 5,0
            1, len(lobes),
            linewidth=2,
            edgecolor='black',
            facecolor='whitesmoke'
        )
    )

    ax_pct.add_patch(
        patches.Rectangle(
            (0, len(lobes)),
            len(bands)+2, 1,
            linewidth=2,
            edgecolor='black',
            facecolor='gainsboro'
        )
    )

    ax_pct.add_patch(
        patches.Rectangle(
            (len(bands), len(lobes)),
            1, 1,
            linewidth=2,
            edgecolor='black',
            facecolor='gainsboro'
        )
    )


    plt.tight_layout()

    return fig, TBR_lobes, TBR_mean
