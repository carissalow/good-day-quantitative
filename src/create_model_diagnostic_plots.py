import pandas as pd 
import numpy as np 
import plotnine as p9

from utils import format_activity_names

def plot_residual_vs_fitted(data, width, height, ncol):
    return (
        p9.ggplot(data=data)
        + p9.geom_point(mapping=p9.aes(x="fitted", y="resid"), alpha=0.5)
        + p9.facet_wrap("activity", scales="free", ncol=ncol)
        + p9.scale_x_continuous(expand=[0, 0])
        + p9.scale_y_continuous(expand=[0, 0])
        + p9.labs(
            x="Fitted values",
            y="Residuals",
            title="Residual vs. fitted plots"
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(width, height),
            plot_title=p9.element_text(face="bold"),
            strip_background=p9.element_rect(fill="white")
        )
    )

def plot_residual_distribution(data, width, height, ncol):
    return (
        p9.ggplot(data=data)
        + p9.geom_density(mapping=p9.aes(x="resid"))
        + p9.facet_wrap("activity", scales="free", ncol=ncol)
        + p9.scale_x_continuous(expand=[0, 0])
        + p9.scale_y_continuous(expand=[0, 0])
        + p9.labs(
            x="Residuals",
            y="Density",
            title="Density plots"
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(width, height),
            plot_title=p9.element_text(face="bold"),
            strip_background=p9.element_rect(fill="white")
        )
    )

def plot_residual_qq(data, width, height, ncol): 
    return (
        p9.ggplot(data=data)
        + p9.geom_abline(color="red", alpha=0.5)
        + p9.geom_point(mapping=p9.aes(x="theoretical_quantiles", y="sample_quantiles"), alpha=0.5)
        + p9.facet_wrap("activity", scales="free", ncol=ncol)
        + p9.scale_x_continuous(expand=[0, 0])
        + p9.scale_y_continuous(expand=[0, 0])
        + p9.labs(
            x="Theoretical quantiles",
            y="Sample quantiles",
            title="Q-Q plots"
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(width, height),
            plot_title=p9.element_text(face="bold"),
            strip_background=p9.element_rect(fill="white")
        )
    )

def main():
    FIG_WIDTH = 15
    FIG_HEIGHT = 10
    FIG_NCOL = 5

    diagnostics = pd.read_csv("output/results/univariable_linear_mixed_models_diagnostics.csv")
    diagnostics["activity"] = format_activity_names(diagnostics["model_formula"].str.replace("^.*~ ", "", regex=True))

    resid_vs_fitted_plot = plot_residual_vs_fitted(diagnostics, width=FIG_WIDTH, height=FIG_HEIGHT, ncol=FIG_NCOL)
    resid_vs_fitted = resid_vs_fitted_plot.draw(show=False)
    resid_vs_fitted.savefig('output/figures/plot_resid_vs_fitted.png', dpi=1000, transparent=False)

    resid_distn_plot = plot_residual_distribution(diagnostics, width=FIG_WIDTH, height=FIG_HEIGHT, ncol=FIG_NCOL)
    resid_distn = resid_distn_plot.draw(show=False)
    resid_distn.savefig('output/figures/plot_resid_distribution.png', dpi=1000, transparent=False)

    resid_qq_plot = plot_residual_qq(diagnostics, width=FIG_WIDTH, height=FIG_HEIGHT, ncol=FIG_NCOL)
    resid_qq = resid_qq_plot.draw(show=False)
    resid_qq.savefig('output/figures/plot_resid_qq.png', dpi=1000, transparent=False)

if __name__ == "__main__":
    main()