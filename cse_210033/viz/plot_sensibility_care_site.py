from functools import reduce
from typing import Dict

import altair as alt
import pandas as pd

from cse_210033.statistical_analysis.utils.supplementary_variables import t_test

from .utils import add_selections


def plot_sensibility_care_site(
    indicators: Dict[str, pd.DataFrame],
    config: Dict[str, str],
    threshold: str = None,
    min_c_0: float = None,
    max_error: float = None,
):
    alt.data_transformers.disable_max_rows()
    stats_config = config["statistical_analysis"]

    # Encoding
    color = alt.Color(
        "Statistical analysis:N",
        title=None,
        sort=["Naive analysis", "Complete-source-only analysis"],
    )

    indicator_charts = []
    indicator_data = []
    selections = None
    for i, outcome_name in enumerate(indicators.keys()):
        indicator = indicators[outcome_name].copy()
        if threshold and "threshold" in indicator.columns:
            indicator = indicator[indicator.threshold == threshold]
        if "max_error" in indicator.columns:
            if max_error is not None:
                indicator = indicator[indicator.max_error == max_error]
            else:
                indicator.max_error = indicator.max_error.replace(
                    indicator.max_error.sort_values(ascending=False).unique(),
                    ["No filter", "Q3", "median", "Q1"],
                )
        if "min_c_0" in indicator.columns:
            if min_c_0 is not None:
                indicator = indicator[indicator.min_c_0 == min_c_0]
            else:
                indicator.min_c_0 = indicator.min_c_0.replace(
                    indicator.min_c_0.sort_values().unique(),
                    ["No filter", "Q1", "median", "Q3"],
                )
        indicator = t_test(indicator, x_col="sub_cohort", y_col="rate")

        indicator_cs_only = indicator[~(indicator.care_site_id == "All")]
        indicator_cs_all = (
            indicator[indicator.care_site_id == "All"]
            .rename(
                columns={
                    "p_value": "p_value_all",
                    "alpha_0": "alpha_0_all",
                    "alpha_1": "alpha_1_all",
                    "mean_value": "mean_value_all",
                }
            )
            .drop(columns="care_site_id")
        )
        index = list(
            {
                "threshold",
                "max_error",
                "start_observation_date",
                "Statistical analysis",
                "min_c_0",
                "outcome_name",
                "young_limit_age",
            }.intersection(indicator.columns)
        )
        indicator = indicator_cs_only.merge(
            indicator_cs_all,
            how="left",
            on=index,
        )
        sens_charts = []
        base = (
            alt.Chart(indicator)
            .encode(
                x=alt.X(
                    "Statistical analysis:N",
                    stack="center",
                    impute=None,
                    title=None,
                    sort=["Naive analysis", "Complete-source-only analysis"],
                    axis=None,
                ),
                color=color,
            )
            .properties(width=75)
        )
        for y_variable in ["alpha_1", "mean_value"]:
            y_titles = {"alpha_1": "Slope", "mean_value": "Mean value"}
            # Diamonds
            points = base.mark_point(
                shape="diamond", stroke="black", filled=True, size=150
            ).encode(
                y="min({}_all):Q".format(y_variable),
            )
            points, selections = add_selections(
                result_chart=points,
                data=indicator,
                selections=selections,
                excluded_selections=["care_site_id"],
            )

            # Box plot
            box_plot = base.mark_boxplot(extent=300).encode(
                y=alt.Y(
                    "{}:Q".format(y_variable),
                    title=y_titles[y_variable],
                    axis=alt.Axis(grid=False),
                    scale=alt.Scale(zero=True),
                ),
            )
            box_plot, selections = add_selections(
                result_chart=box_plot,
                data=indicator,
                selections=selections,
                excluded_selections=["care_site_id"],
            )

            if y_variable == "alpha_1":
                centrer_line = (
                    alt.Chart(pd.DataFrame({"y": [0]}))
                    .mark_rule(strokeWidth=2)
                    .encode(y="y")
                )
                sens_chart = alt.layer(box_plot + points + centrer_line)
            else:
                sens_chart = alt.layer(box_plot + points)

            sens_chart = sens_chart.facet(
                column=alt.Column(
                    "yearmonth(start_observation_date):T",
                    header=alt.Header(
                        titleOrient="left",
                        title=stats_config[outcome_name]["title"]
                        if y_variable == "alpha_1"
                        else "",
                        titleFontSize=20,
                        labelFontSize=19,
                        labelFontWeight="bold",
                        labels=i == 0,
                    ),
                ),
                spacing=12,
            )
            sens_charts.append(sens_chart)
        indicator_chart = reduce(
            lambda chart_1, chart_2: alt.hconcat(chart_1, chart_2, spacing=60),
            sens_charts,
        )
        indicator["outcome_name"] = stats_config[outcome_name]["event_name"]
        indicator_charts.append(indicator_chart)
        indicator_data.append(indicator)

    sens_cs_data = pd.concat(indicator_data)
    sens_cs_chart = reduce(
        lambda chart_1, chart_2: alt.vconcat(chart_1, chart_2, spacing=5),
        indicator_charts,
    )
    sens_cs_chart = (
        sens_cs_chart.configure_legend(
            labelFontSize=19,
            title=None,
            orient="top",
            symbolOpacity=1,
            symbolType="square",
            symbolSize=300,
            labelLimit=500,
        )
        .configure_axis(
            labelFontSize=19,
            titleFontSize=20,
            titleFontStyle="italic",
        )
        .configure_view(strokeWidth=3)
    )
    return sens_cs_chart, sens_cs_data
