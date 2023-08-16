from functools import reduce
from typing import Dict

import altair as alt
import pandas as pd

from .utils import (
    CARE_SITE_LEVELS,
    add_covid_band,
    add_cs_number,
    add_linear_test,
    add_selections,
    add_x_translation,
)


def plot_quality_indicators(
    quality_indicators: Dict[str, pd.DataFrame],
    config: Dict[str, str],
    with_cs_number: bool = True,
    icu_box_position: bool = False,
    cs_count_summary: pd.DataFrame = None,
    with_covid19_band: bool = False,
    no_care_site: bool = True,
    no_MCD: bool = True,
    with_test: bool = False,
    show_zero: bool = True,
):
    selections = None
    bar_chart = None
    quality_charts = []
    quality_data = []
    stats_config = config["statistical_analysis"]
    if with_cs_number and cs_count_summary is not None:
        cs_count_summary = cs_count_summary.replace(
            {"care_site_level": CARE_SITE_LEVELS}
        )
        cs_count_summary["cs_considered"] = (
            cs_count_summary["cso_cs_count"].astype(str)
            + " ("
            + (
                cs_count_summary["cso_cs_count"]
                / cs_count_summary["naive_cs_count"]
                * 100
            ).map("{:,.1f}".format)
            + " %)"
        )
        cs_count_summary["care_site_level"] = (
            cs_count_summary["care_site_level"].astype(str) + "s"
        )
        cs_count_summary["cs_considered"] = cs_count_summary[
            ["cs_considered", "care_site_level"]
        ].values.tolist()
        cs_count_summary = cs_count_summary[
            [
                "start_observation_date",
                "max_error",
                "min_c_0",
                "cs_considered",
                "outcome_name",
            ]
        ]
    for i, outcome_name in enumerate(quality_indicators.keys()):
        indicator = quality_indicators[outcome_name].copy()
        if with_cs_number and cs_count_summary is not None:
            indicator = (
                cs_count_summary[cs_count_summary.outcome_name == outcome_name]
                .drop(columns="outcome_name")
                .merge(indicator, on=["start_observation_date", "max_error", "min_c_0"])
            )
            indicator["text_x"] = "2020-06-01"
        # Pre-processing
        if no_MCD and "MCD" in indicator.columns:
            indicator = indicator[indicator.MCD == "00 ALL"].drop(columns=["MCD"])
        if no_care_site and "care_site_id" in indicator.columns:
            indicator = indicator[indicator.care_site_id == "All"].drop(
                columns=["care_site_id"]
            )
        if "max_error" in indicator.columns:
            max_errors = indicator.max_error.sort_values(ascending=False).unique()
            indicator.max_error = indicator.max_error.replace(
                max_errors, ["No filter", "Q3", "median", "Q1"]
            )
        if "min_c_0" in indicator.columns:
            min_c_0s = indicator.min_c_0.sort_values().unique()
            indicator.min_c_0 = indicator.min_c_0.replace(
                min_c_0s, ["No filter", "Q1", "median", "Q3"]
            )
        indicator = indicator.drop(columns=["n_events"])
        indicator["date_to_filter"] = indicator["sub_cohort"].copy()
        indicator["legend_predictor"] = "Quality indicator"
        indicator["legend_model"] = "Linear model"
        indicator["start_observation_date"] = indicator.start_observation_date.astype(
            "datetime64[ns]"
        ).dt.strftime("Starting date: %b %Y")
        x_axis = i == (len(quality_indicators) - 1)
        legend = i == 0
        header_labels = i == 0
        title = stats_config[outcome_name]["title"]
        # Plot
        chart, data, selections = plot_quality_indicator(
            indicator,
            y_title=title,
            legend=legend,
            header_labels=header_labels,
            x_axis=x_axis,
            with_cs_number=with_cs_number,
            icu_box_position=icu_box_position,
            with_covid19_band=with_covid19_band,
            with_test=with_test,
            show_zero=show_zero,
            selections=selections,
        )
        if i == 0 and not no_care_site and "care_site_id" in indicator.columns:
            bar_chart = (
                alt.Chart(indicator)
                .mark_bar()
                .encode(
                    x=alt.X("care_site_id:N", title="Care site id", sort="-y"),
                    y=alt.Y("sum(n_total):Q", title="No. recorded hospitalizations"),
                    color=alt.condition(
                        selections["care_site_id"],
                        alt.Color(
                            "care_site_id:N",
                            legend=None,
                            sort={
                                "field": "n_total",
                                "op": "sum",
                                "order": "descending",
                            },
                        ),
                        alt.value("lightgray"),
                    ),
                )
            )
            for column, selection in selections.items():
                if column in indicator.columns and column != "care_site_id":
                    bar_chart = bar_chart.transform_filter(selection)
        data["outcome_name"] = outcome_name
        quality_data.append(data)
        quality_charts.append(chart)
    quality_data = pd.concat(quality_data)
    quality_charts = reduce(
        lambda chart_1, chart_2: alt.vconcat(
            chart_1, chart_2, spacing=20
        ).resolve_scale(color="independent"),
        quality_charts,
    )
    if bar_chart:
        quality_charts = alt.vconcat(
            quality_charts, bar_chart, spacing=5
        ).resolve_scale(color="independent")
    # Configuration
    quality_charts = (
        quality_charts.configure_axis(
            labelFontSize=19, titleFontSize=20, labelLimit=500
        )
        .configure_legend(
            labelFontSize=20,
            symbolSize=500,
            symbolStrokeWidth=3,
            labelLimit=500,
        )
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=20)
    )
    return quality_charts, quality_data


def plot_quality_indicator(
    data: Dict[str, pd.DataFrame],
    y_title: str,
    with_cs_number: bool,
    icu_box_position: bool,
    with_covid19_band: bool,
    with_test: bool,
    show_zero: bool,
    header_labels: bool,
    x_axis: bool,
    legend: bool,
    selections: Dict[str, alt.SelectionParameter],
):
    alt.data_transformers.disable_max_rows()

    # Encoding
    legend_colors = alt.Legend(orient="top") if legend else None
    legend_strokeDash = (
        alt.Legend(
            symbolType="stroke",
            symbolStrokeColor="grey",
            orient="top",
        )
        if legend
        else None
    )
    legend_strokeWidth = (
        alt.Legend(
            symbolType="stroke",
            symbolStrokeColor="grey",
            symbolDash=[4, 4],
            orient="top",
        )
        if legend
        else None
    )
    colors = alt.Color(
        "Statistical analysis:N",
        title=None,
        legend=legend_colors,
        sort=["Naive analysis", "Complete-source-only analysis"],
    )
    charts = []
    for i, start_observation_date in enumerate(data.start_observation_date.unique()):
        if icu_box_position:
            y_position = 70 if i == 0 else 130
        else:
            y_position = 120
        data_observed = data[
            data.start_observation_date == start_observation_date
        ].drop(columns="start_observation_date")

        y_axis = (
            alt.Axis(title=y_title, format=".2f")
            if i == 0
            else alt.Axis(
                labels=False, title="", ticks=False, domain=False, format=".2f"
            )
        )
        # Base chart
        base = alt.Chart(
            data_observed, title=start_observation_date if header_labels else ""
        ).properties(
            width=(data_observed.sub_cohort.max() - data_observed.sub_cohort.min()).days
            * 0.17,
            height=200,
        )

        base_main = base.encode(
            x=alt.X(
                "sub_cohort:T",
                axis=alt.Axis(
                    tickCount="year",
                    format="%Y",
                    title="Admission date",
                    labelExpr="datum.label % 2 == 0 ? datum.label : ''",
                    labelSeparation=10,
                )
                if x_axis
                else alt.Axis(
                    tickCount="year",
                    ticks=False,
                    title=None,
                    labelFlush=False,
                    labels=False,
                ),
            ),
            color=colors,
        )

        # generate the points
        points = base_main.mark_line().encode(
            y=alt.Y("rate:Q", scale=alt.Scale(zero=show_zero, domainMin=0)),
            strokeDash=alt.StrokeDash(
                "legend_predictor",
                title="",
                legend=legend_strokeDash,
            ),
        )

        # generate the regression
        reg = (
            base_main.mark_line(strokeDash=[5, 5], clip=True)
            .transform_regression(
                on="sub_cohort",
                regression="rate",
                groupby=[
                    "Statistical analysis",
                    "legend_model",
                ],
            )
            .encode(
                y=alt.Y("rate:Q", axis=y_axis, scale=alt.Scale(domainMin=0)),
                strokeWidth=alt.StrokeWidth(
                    "legend_model",
                    title="",
                    legend=legend_strokeWidth,
                ),
            )
        )
        chart = alt.layer(points, reg)

        # Cs number considered with CSO method
        if with_cs_number:
            box, text = add_cs_number(base=base, y_position=y_position)
            chart = alt.layer(chart, text)

        # X translation
        chart = add_x_translation(result_chart=chart)

        # Filter selections
        chart, selections = add_selections(
            result_chart=chart, data=data, selections=selections
        )

        # Add box
        if with_cs_number:
            chart = alt.layer(box, chart)

        # Covid 19 period
        if with_covid19_band:
            chart = add_covid_band(result_chart=chart)

        charts.append(chart)
    result_chart = reduce(
        lambda chart_1, chart_2: alt.hconcat(chart_1, chart_2).resolve_scale(
            y="shared"
        ),
        charts,
    )

    # Test
    if with_test:
        result_chart = add_linear_test(
            result_chart=result_chart,
            data=data,
            x_col="sub_cohort",
            y_col="rate",
            selections=selections,
        )

    return result_chart, data, selections
