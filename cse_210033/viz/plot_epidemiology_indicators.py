from datetime import datetime
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


def plot_epidemiology_indicators(
    epidemiology_indicators: Dict[str, pd.DataFrame],
    config: Dict[str, str],
    with_cs_number: bool = True,
    cs_count_summary: pd.DataFrame = None,
    no_care_site: bool = True,
    no_MCD: bool = True,
    with_covid19_band: bool = True,
    with_test: bool = False,
    with_model: bool = False,
    show_zero: bool = True,
):
    selections = None
    bar_chart = None
    epidemiology_charts = []
    epidemiology_data = []
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
    for i, outcome_name in enumerate(epidemiology_indicators.keys()):
        indicator = epidemiology_indicators[outcome_name].copy()
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
        indicator["date_to_filter"] = indicator["sub_cohort"].copy()
        indicator["legend_predictor"] = "Data points"
        indicator["legend_model"] = "Linear model"
        indicator["start_observation_date"] = indicator.start_observation_date.astype(
            "datetime64[ns]"
        ).dt.strftime("Starting date: %b %Y")
        x_axis = i == (len(epidemiology_indicators) - 1)
        legend = i == 0
        header_labels = i == 0
        title = stats_config[outcome_name]["title"]
        chart, data, selections = plot_epidemiology_indicator(
            indicator,
            y_title=title,
            legend=legend,
            x_axis=x_axis,
            header_labels=header_labels,
            with_cs_number=with_cs_number,
            with_covid19_band=with_covid19_band,
            with_test=with_test,
            with_model=with_model,
            show_zero=show_zero,
            selections=selections,
        )
        data["outcome_name"] = outcome_name
        epidemiology_data.append(data)
        epidemiology_charts.append(chart)
        if i == 0 and not no_care_site and "care_site_id" in indicator.columns:
            bar_chart = (
                alt.Chart(indicator)
                .mark_bar()
                .encode(
                    x=alt.X("care_site_id:N", title="Care site id", sort="-y"),
                    y=alt.Y("sum(n_events):Q", title="No. recorded condition"),
                    color=alt.condition(
                        selections["care_site_id"],
                        alt.Color(
                            "care_site_id:N",
                            legend=None,
                            sort={
                                "field": "n_events",
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
    epidemiology_data = pd.concat(epidemiology_data)
    epidemiology_charts = reduce(
        lambda chart_1, chart_2: alt.vconcat(
            chart_1, chart_2, spacing=20
        ).resolve_scale(color="independent"),
        epidemiology_charts,
    )
    if bar_chart:
        epidemiology_charts = alt.vconcat(
            epidemiology_charts, bar_chart, spacing=5
        ).resolve_scale(color="independent")
    # Configuration
    epidemiology_charts = (
        epidemiology_charts.configure_axis(
            labelFontSize=19, titleFontSize=20, labelLimit=500
        )
        .configure_legend(
            labelFontSize=20,
            symbolSize=500,
            symbolStrokeWidth=3,
            labelLimit=500,
        )
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=20, dy=-10)
    )
    return epidemiology_charts, epidemiology_data


def plot_epidemiology_indicator(
    data: Dict[str, pd.DataFrame],
    y_title: str,
    with_cs_number: bool,
    with_covid19_band: bool,
    with_test: bool,
    with_model: bool,
    header_labels: bool,
    show_zero: bool,
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
        if legend and with_model
        else None
    )
    legend_strokeWidth = (
        alt.Legend(
            symbolType="stroke",
            symbolStrokeColor="grey",
            symbolDash=[2, 2],
            orient="top",
        )
        if legend and with_model
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
        data_observed = data[
            data.start_observation_date == start_observation_date
        ].drop(columns="start_observation_date")
        min_sub_cohort = (
            data_observed.groupby(["max_error", "min_c_0", "Statistical analysis"])
            .agg({"sub_cohort": "min"})
            .sub_cohort.max()
        )
        data_observed["sub_cohort"] = data_observed["sub_cohort"].mask(
            data_observed["sub_cohort"] <= min_sub_cohort,
            datetime((data_observed.sub_cohort.dt.year.min()), 1, 1),
        )
        y_axis = (
            alt.Axis(title=y_title)
            if i == 0
            else alt.Axis(labels=False, title="", ticks=False, domain=False)
        )
        # Base chart
        base = alt.Chart(
            data_observed,
            title=start_observation_date if header_labels else "",
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

        # Data points
        points = base_main.mark_line().encode(
            y=alt.Y(
                "n_events:Q", axis=y_axis, scale=alt.Scale(zero=show_zero, domainMin=0)
            ),
            strokeDash=alt.StrokeDash(
                "legend_predictor",
                title="",
                legend=legend_strokeDash,
            ),
        )

        # Linear regression
        if with_model:
            # Per-winter max points
            points_max = base_main.mark_point(
                filled=False, size=200, shape="diamond"
            ).encode(
                y=alt.Y(
                    "max_events:Q",
                    axis=y_axis,
                    scale=alt.Scale(zero=show_zero, domainMin=0),
                )
            )
            reg = (
                base_main.mark_line(strokeDash=[5, 5], clip=True)
                .encode(
                    y=alt.Y("max_events:Q", axis=y_axis, scale=alt.Scale(domainMin=0)),
                    strokeWidth=alt.StrokeWidth(
                        "legend_model",
                        title="",
                        legend=legend_strokeWidth,
                    ),
                )
                .transform_regression(
                    "sub_cohort",
                    "max_events",
                    groupby=[
                        "Statistical analysis",
                        "start_observation_date",
                        "legend_model",
                    ],
                )
            )
            chart = alt.layer(points, points_max, reg)

        else:
            chart = alt.layer(points)

        # Cs number considered with CSO method
        if with_cs_number:
            box, text = add_cs_number(base=base, y_position=-10)
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
            x_col="school_years",
            y_col="max_events",
            selections=selections,
        )

    return result_chart, data, selections
