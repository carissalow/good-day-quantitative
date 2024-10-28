import pandas as pd 
import numpy as np 
import plotnine as p9

from utils import format_activity_names

def plot_residual_qq(data, width, height, ncol): 
    return (
        p9.ggplot(data=data)
        + p9.geom_abline(color="red", alpha=0.8)
        + p9.geom_point(mapping=p9.aes(x="theoretical_quantiles", y="sample_quantiles"), alpha=0.3)
        + p9.facet_wrap("activity", scales="free", ncol=ncol)
        + p9.scale_x_continuous(expand=[0.01, 0.01])
        + p9.scale_y_continuous(expand=[0.01, 0.01])
        + p9.labs(
            x="Theoretical quantiles",
            y="Sample quantiles",
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(width, height),
            plot_title=p9.element_text(face="bold"),
            strip_background=p9.element_rect(fill="white"),
            strip_text=p9.element_text(size=8)
        )
    )

def main():
    FIG_WIDTH = 12
    FIG_HEIGHT = 14
    FIG_NCOL = 4

    diagnostics = pd.read_csv("output/results/univariable_linear_mixed_models_diagnostics.csv")
    diagnostics["activity"] = format_activity_names(diagnostics["model_formula"].str.replace("^.*~ ", "", regex=True))  

    supp_fig_1_plot = plot_residual_qq(diagnostics, width=FIG_WIDTH, height=FIG_HEIGHT, ncol=FIG_NCOL)
    supp_fig_1 = supp_fig_1_plot.draw(show=False)
    supp_fig_1.savefig('output/figures/supplemental_figure_1.png', dpi=1000, transparent=False)

if __name__ == "__main__":
    main()