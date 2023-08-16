from typing import Dict, List

import altair as alt
import pandas as pd

from cse_210033.statistical_analysis.utils.supplementary_variables import t_test

CARE_SITE_LEVELS = {
    "Hôpital": "hospital",
    "Unité Fonctionnelle (UF)": "department",
    "Unité d’hébergement (UH)": "unit",
}


def add_covid_band(result_chart: alt.Chart):
    # Covid 19 censorship
    area = (
        alt.Chart(
            pd.DataFrame(
                {
                    "covid_start_date": ["2020-03-01"],
                    "Censorship": ["Start of COVID-19 epidemic"],
                }
            )
        )
        .mark_rule(color="black", strokeWidth=3)
        .encode(
            x="covid_start_date:T",
            opacity=alt.Opacity(
                "Censorship:N",
                scale=alt.Scale(rangeMin=0.5),
                legend=alt.Legend(orient="top", title=""),
            ),
        )
    )
    return alt.layer(result_chart, area)


def add_cs_number(base: alt.Chart, y_position: float):
    # Add number of care site considered for CSO method

    text = base.mark_text(
        align="left", baseline="top", color="#f58518", fontSize=19
    ).encode(
        x="text_x:T",
        y=alt.value(y_position),  # pixels from top
        text="min(cs_considered):N",
    )

    box = (
        alt.Chart(
            pd.DataFrame(
                {
                    "box_x": ["2020-05-01"],
                    "box_x2": ["2022-05-01"],
                }
            )
        )
        .mark_rect(stroke="black", color="white")
        .encode(
            x="box_x:T",
            x2="box_x2:T",
            y=alt.value(y_position - 2),
            y2=alt.value(y_position + 45),
        )
    )
    return box, text


def add_linear_test(
    result_chart: alt.Chart,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    selections: Dict[str, alt.SelectionParameter],
):
    result_test = t_test(data, x_col=x_col, y_col=y_col, as_string=True)
    result_test = result_test.melt(
        id_vars=set(result_test.columns) - {"p_value", "alpha_0", "alpha_1"},
        var_name="model",
    )
    test = (
        alt.Chart(result_test)
        .mark_text(align="center", baseline="middle", color="black")
        .encode(
            x=alt.X("model:N", axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Statistical analysis:N", axis=alt.Axis(title=None)),
            column=alt.Column(
                "yearmonth(start_observation_date):T",
                header=alt.Header(title=None, labels=False),
            ),
            text="value:N",
        )
    ).properties(width=800)
    for column, selection in selections.items():
        if column in result_test.columns:
            test = test.transform_filter(selection)
    return alt.vconcat(test, result_chart, center=True).resolve_scale(
        color="independent", strokeDash="independent", strokeWidth="independent"
    )


def add_selections(
    result_chart: alt.Chart,
    data: pd.DataFrame,
    selections: Dict[str, alt.SelectionParameter] = None,
    excluded_selections: List[str] = None,
):
    if excluded_selections is None:
        excluded_selections = []
    if selections is None:
        selections = {}
        # Min c0 selection
        if "min_c_0" in data.columns and "min_c_0" not in excluded_selections:
            options = list(data.min_c_0.unique())
            if data.max_error.dtype == float:
                nums = list(round(data.min_c_0, 3).astype(str).unique())
                params = ["No filter :", "Q1 :", "Median :", "Q3 :"]
                labels = [
                    "{} {}".format(param, num) for num, param in zip(nums, params)
                ]
                init = 0
            else:
                labels = options
                init = "No filter"

            min_c_0_dropdown = alt.binding_select(
                options=options,
                labels=labels,
                name="Min c_0 : ",
            )
            min_c_0_selection = alt.selection_point(
                fields=["min_c_0"],
                bind=min_c_0_dropdown,
                value=init,
            )
            selections["min_c_0"] = min_c_0_selection
            result_chart = result_chart.add_params(min_c_0_selection)

        # Max error selection
        if "max_error" in data.columns and "max_error" not in excluded_selections:
            options = list(data.max_error.unique())
            if data.max_error.dtype == float:
                nums = list(round(data.max_error, 3).astype(str).unique())
                params = ["No filter :", "Q3 :", "Median :", "Q1 :"]
                labels = [
                    "{} {}".format(param, num) for num, param in zip(nums, params)
                ]
                init = max(options)
            else:
                labels = options
                init = "No filter"

            max_error_dropdown = alt.binding_select(
                options=options,
                labels=labels,
                name="Max error : ",
            )
            max_error_selection = alt.selection_point(
                fields=["max_error"],
                bind=max_error_dropdown,
                value=init,
            )
            selections["max_error"] = max_error_selection
            result_chart = result_chart.add_params(max_error_selection)

        # Young age limit selection
        if (
            "young_limit_age" in data.columns
            and "young_limit_age" not in excluded_selections
        ):
            options = list(data.young_limit_age.unique())
            young_limit_age_dropdown = alt.binding_select(
                options=labels,
                name="Age limit : ",
            )
            young_limit_age_selection = alt.selection_point(
                fields=["young_limit_age"],
                bind=young_limit_age_dropdown,
                value=min(options),
            )
            selections["young_limit_age"] = young_limit_age_selection
            result_chart = result_chart.add_params(young_limit_age_selection)

        # MCD selection
        if "MCD" in data.columns and "MCD" not in excluded_selections:
            options = list(data.MCD.sort_values().unique())
            mcd_dropdown = alt.binding_select(
                options=options,
                name="MCD : ",
            )
            mcd_selection = alt.selection_point(
                fields=["MCD"],
                bind=mcd_dropdown,
                value=min(options),
            )
            selections["MCD"] = mcd_selection
            result_chart = result_chart.add_params(mcd_selection)

        # Threshold selection
        if "threshold" in data.columns and "threshold" not in excluded_selections:
            options = list(data.threshold.unique())
            threshold_dropdown = alt.binding_select(
                options=options,
                name="Threshold : ",
            )
            threshold_selection = alt.selection_point(
                fields=["threshold"],
                bind=threshold_dropdown,
                value=min(options),
            )
            selections["threshold"] = threshold_selection
            result_chart = result_chart.add_params(threshold_selection)

        # Event selection
        if "outcome_name" in data.columns and "outcome_name" not in excluded_selections:
            options = list(data.event.unique())
            outcome_dropdown = alt.binding_select(
                options=options,
                name="Outcome : ",
            )
            outcome_selection = alt.selection_point(
                fields=["outcome_name"],
                bind=outcome_dropdown,
                value=min(options),
            )
            selections["outcome_name"] = outcome_selection
            result_chart = result_chart.add_params(outcome_selection)

        # Care site selection
        if "care_site_id" in data.columns and "care_site_id" not in excluded_selections:
            options = list(data.care_site_id.unique())
            care_site_dropdown = alt.binding_select(
                options=options,
                name="Care site id : ",
            )
            care_site_selection = alt.selection_point(
                fields=["care_site_id"],
                bind=care_site_dropdown,
                value="All",
            )
            selections["care_site_id"] = care_site_selection
            result_chart = result_chart.add_params(care_site_selection)
    for column, selection in selections.items():
        if column in data.columns:
            result_chart = result_chart.transform_filter(selection)
    return result_chart, selections


def add_x_translation(result_chart: alt.Chart):
    # X translation
    t_0_slider = alt.binding(
        input="start_date",
        name="Start date : ",
    )
    t_0_selection = alt.selection_point(
        name="start_date",
        fields=["start_date"],
        bind=t_0_slider,
        value="2013-01-01",
    )
    result_chart = result_chart.add_params(t_0_selection).transform_filter(
        alt.datum.date_to_filter >= t_0_selection.start_date
    )
    return result_chart
