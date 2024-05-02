import pandas as pd 
import numpy as np
import itertools
import plotnine as p9
from utils import format_activity_names

def create_correlation_heatmap(data, x="", y="", fill="r", x_label="", y_label="", fig_width=7.5, fig_height=6):
    """
    Create a heatmap figure displaying correlation coefficient on the z-axis (i.e., fill color)
    Parameters:
        data (pandas.DataFrame): Long-format data
        x (str): Name of the variable to plot on the x-axis
        y (str): Name of the variable to plot on the y-axis
        fill (str): Name of the variable to plot on the z-axis. Default is "r"
        x_label (str): Title for the x-axis
        y_label (str): Title for the y-axis
        fig_width (numeric): Width of the figure in inches. Default is 7.5
        fig_height (numeric): Height of the figure in inches. Default is 6
    Returns:
        A figure which can be saved to a png file with `plotnine.ggsave()`
    """
    return (
        p9.ggplot(data) 
            + p9.geom_tile(p9.aes(y=y, x=x, fill=fill), color="black", size=0.05)
            + p9.scale_x_discrete(expand=(0, 0))
            # reverse y-axis tick label order so activities are in descending alphabetical order
            + p9.scale_y_discrete(expand=(0, 0), limits=data.sort_values(y)[y].unique()[::-1])
            + p9.scale_fill_cmap(cmap_name="RdBu", breaks=(-0.99, -0.5, 0, 0.5, 0.99), na_value="grey")
            + p9.coord_equal() 
            + p9.guides(fill=p9.guide_colorbar(draw_ulim=False, draw_llim=False))
            + p9.labs(x=x_label, y=y_label, fill="Correlation\ncoefficient")
            + p9.theme_light()
            + p9.theme(
                axis_text_x=p9.element_text(size=8),
                axis_text_y=p9.element_text(size=10),
                axis_title=p9.element_text(size=12),
                legend_title=p9.element_text(size=10),
                legend_ticks_length=0,
                legend_position="right",
                figure_size=(fig_width, fig_height)
            )
    )

def main():
    within_person_correlations = pd.read_csv("output/results/within_person_point_biserial_correlations.csv")

    pids = within_person_correlations["pid"].unique()
    activities = within_person_correlations["activity"].unique()
    pid_activity_combinations = pd.DataFrame(columns=["pid", "activity"], data=list(itertools.product(*[pids, activities])))
    
    within_person_correlations = within_person_correlations.merge(pid_activity_combinations, how="right")
    within_person_correlations["pid"] = within_person_correlations["pid"].astype("category")
    within_person_correlations["activity"] = format_activity_names(within_person_correlations["activity"])

    heatmap = create_correlation_heatmap(
        data=within_person_correlations,
        x="pid", 
        y="activity", 
        fill="r",
        x_label="Participant ID", 
        y_label="Activity",
        fig_width=7.5, 
        fig_height=6
    )

    p9.ggsave(heatmap, filename="output/figures/figure_1.png", dpi=1000)

if __name__ == "__main__":
    main()