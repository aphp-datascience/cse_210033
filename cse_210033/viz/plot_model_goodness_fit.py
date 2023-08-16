from functools import reduce
from typing import Dict

import altair as alt
import pandas as pd
import polars as pl
from edsteva.models.step_function import StepFunction
from edsteva.probes import ConditionProbe, NoteProbe, VisitProbe
from edsteva.viz.plots import normalized_probe_plot
from edsteva.viz.utils import month_diff

from cse_210033.statistical_analysis.utils.complete_source import filter_estimates


def plot_model_goodness_fit(
    visit_probe: VisitProbe,
    note_probe: NoteProbe,
    note_probe_per_visit: NoteProbe,
    condition_probe_per_visit: ConditionProbe,
    visit_model: StepFunction,
    note_model: StepFunction,
    note_per_visit_model: StepFunction,
    condition_per_visit_model: StepFunction,
    config: Dict,
    t_min: int = -50,
    t_max: int = 50,
    min_c_0: float = 0.15,
):
    alt.data_transformers.disable_max_rows()
    stats_config = config["statistical_analysis"]
    (
        probe_line_config,
        error_line_config,
        model_line_config,
    ) = get_normalized_line_config(with_legend=True)
    estimates_selections, estimates_filters = get_selections(min_c_0=min_c_0)
    charts = {}
    data = []
    # Hospitalization
    visit_probe.predictor = (
        filter_estimates(
            ehr_estimates=pl.from_pandas(visit_probe.predictor),
            **stats_config["hospit_visit"]
        )
        .to_pandas()
        .replace({"care_site_level": {"Hôpital": "1 - Hospitals"}})
    )
    visit_model.estimates = visit_model.estimates.replace(
        {"care_site_level": {"Hôpital": "1 - Hospitals"}}
    )
    charts["hospit_visit"] = normalized_probe_plot(
        probe=visit_probe,
        fitted_model=visit_model,
        model_line_config=model_line_config,
        probe_line_config=probe_line_config,
        error_line_config=error_line_config,
        main_chart_config=get_main_chart_config(
            with_X_title=False, Y_title="Hospitalizations"
        ),
        estimates_selections=estimates_selections,
        estimates_filters=estimates_filters,
        chart_style=False,
        t_min=t_min,
        t_max=t_max,
        indexes_to_remove=[
            "care_site_id",
            "care_site_short_name",
            "specialties_set",
            "stay_type",
        ],
    )
    visit_probe.predictor["EHR_functionality"] = "Hospitalization"
    data.append(
        visit_probe.predictor.merge(visit_model.estimates, on=visit_probe._index)
    )
    visit_probe.reset_predictor()
    visit_model.reset_estimates()

    # Emergency
    (
        probe_line_config,
        error_line_config,
        model_line_config,
    ) = get_normalized_line_config(with_legend=False)
    visit_probe.predictor = (
        filter_estimates(
            ehr_estimates=pl.from_pandas(visit_probe.predictor),
            **stats_config["emergency_visit"]
        )
        .to_pandas()
        .replace({"care_site_level": {"Hôpital": "1 - Hospitals"}})
    )
    visit_model.estimates = visit_model.estimates.replace(
        {"care_site_level": {"Hôpital": "1 - Hospitals"}}
    )
    charts["emergency_visit"] = normalized_probe_plot(
        probe=visit_probe,
        fitted_model=visit_model,
        model_line_config=model_line_config,
        probe_line_config=probe_line_config,
        error_line_config=error_line_config,
        main_chart_config=get_main_chart_config(
            with_X_title=False, Y_title="Emergency stays"
        ),
        estimates_selections=estimates_selections,
        estimates_filters=estimates_filters,
        chart_style=False,
        t_min=t_min,
        t_max=t_max,
        indexes_to_remove=[
            "care_site_id",
            "care_site_short_name",
            "specialties_set",
            "stay_type",
        ],
    )
    visit_probe.predictor["EHR_functionality"] = "Emergency stays"
    data.append(
        visit_probe.predictor.merge(visit_model.estimates, on=visit_probe._index)
    )
    visit_probe.reset_predictor()
    visit_model.reset_estimates()

    # Consultation Note
    note_probe.predictor = (
        filter_estimates(
            ehr_estimates=pl.from_pandas(note_probe.predictor),
            **stats_config["consultation_note"]
        )
        .rename({"detail_care_site_id": "care_site_id"})
        .to_pandas()
        .replace({"care_site_level": {"Unité Fonctionnelle (UF)": "2 - Departments"}})
    )
    note_model.estimates = note_model.estimates.replace(
        {"care_site_level": {"Unité Fonctionnelle (UF)": "2 - Departments"}}
    )
    charts["consultation_note"] = normalized_probe_plot(
        probe=note_probe,
        fitted_model=note_model,
        model_line_config=model_line_config,
        probe_line_config=probe_line_config,
        error_line_config=error_line_config,
        main_chart_config=get_main_chart_config(
            with_X_title=False, Y_title="Consultation reports"
        ),
        chart_style=False,
        estimates_selections=estimates_selections,
        estimates_filters=estimates_filters,
        t_min=t_min,
        t_max=t_max,
        indexes_to_remove=[
            "care_site_id",
            "care_site_short_name",
            "note_type",
            "stay_type",
        ],
    )
    note_probe.predictor["EHR_functionality"] = "Consultation reports"
    data.append(note_probe.predictor.merge(note_model.estimates, on=note_probe._index))
    note_probe.reset_predictor()
    note_model.reset_estimates()

    # Prescription Note
    note_probe_per_visit.predictor = (
        filter_estimates(
            ehr_estimates=pl.from_pandas(note_probe_per_visit.predictor),
            **stats_config["prescription_note"]
        )
        .rename({"detail_care_site_id": "care_site_id"})
        .to_pandas()
        .replace({"care_site_level": {"Unité Fonctionnelle (UF)": "2 - Departments"}})
    )
    note_per_visit_model.estimates = note_per_visit_model.estimates.replace(
        {"care_site_level": {"Unité Fonctionnelle (UF)": "2 - Departments"}}
    )
    charts["prescription_note"] = normalized_probe_plot(
        probe=note_probe_per_visit,
        fitted_model=note_per_visit_model,
        model_line_config=model_line_config,
        probe_line_config=probe_line_config,
        error_line_config=error_line_config,
        main_chart_config=get_main_chart_config(
            with_X_title=False, Y_title="Prescription reports"
        ),
        chart_style=False,
        estimates_selections=estimates_selections,
        estimates_filters=estimates_filters,
        t_min=t_min,
        t_max=t_max,
        indexes_to_remove=[
            "care_site_id",
            "care_site_short_name",
            "note_type",
            "stay_type",
        ],
    )
    note_probe_per_visit.predictor["EHR_functionality"] = "Prescription reports"
    data.append(
        note_probe_per_visit.predictor.merge(
            note_per_visit_model.estimates, on=note_probe_per_visit._index
        )
    )
    note_probe_per_visit.reset_predictor()
    note_per_visit_model.reset_estimates()

    # Diagnostic codes
    condition_probe_per_visit.predictor = (
        filter_estimates(
            ehr_estimates=pl.from_pandas(condition_probe_per_visit.predictor),
            **stats_config["bronchiolitis_condition"]
        )
        .rename({"detail_care_site_id": "care_site_id"})
        .to_pandas()
        .replace({"care_site_level": {"Unité Fonctionnelle (UF)": "2 - Departments"}})
    )
    condition_per_visit_model.estimates = condition_per_visit_model.estimates.replace(
        {"care_site_level": {"Unité Fonctionnelle (UF)": "2 - Departments"}}
    )
    charts["diagnostic_codes"] = normalized_probe_plot(
        probe=condition_probe_per_visit,
        fitted_model=condition_per_visit_model,
        model_line_config=model_line_config,
        probe_line_config=probe_line_config,
        error_line_config=error_line_config,
        main_chart_config=get_main_chart_config(
            with_X_title=False, Y_title="Diagnostic codes"
        ),
        chart_style=False,
        estimates_selections=estimates_selections,
        estimates_filters=estimates_filters,
        t_min=t_min,
        t_max=t_max,
        indexes_to_remove=[
            "care_site_id",
            "care_site_short_name",
            "stay_type",
            "condition_type",
            "diag_type",
            "source_system",
        ],
    )
    condition_probe_per_visit.predictor["EHR_functionality"] = "Diagnostic codes"
    data.append(
        condition_probe_per_visit.predictor.merge(
            condition_per_visit_model.estimates, on=condition_probe_per_visit._index
        )
    )
    condition_probe_per_visit.reset_predictor()
    condition_per_visit_model.reset_estimates()

    # ICU
    visit_probe.predictor = (
        filter_estimates(
            ehr_estimates=pl.from_pandas(visit_probe.predictor),
            **stats_config["icu_visit"]
        )
        .rename({"detail_care_site_id": "care_site_id"})
        .to_pandas()
        .replace({"care_site_level": {"Unité d’hébergement (UH)": "3 - Units"}})
    )
    visit_model.estimates = visit_model.estimates.replace(
        {"care_site_level": {"Unité d’hébergement (UH)": "3 - Units"}}
    )
    charts["icu_visit"] = normalized_probe_plot(
        probe=visit_probe,
        fitted_model=visit_model,
        model_line_config=model_line_config,
        probe_line_config=probe_line_config,
        error_line_config=error_line_config,
        main_chart_config=get_main_chart_config(
            with_X_title=True, Y_title="Intensive care stays"
        ),
        estimates_selections=estimates_selections,
        estimates_filters=estimates_filters,
        chart_style=False,
        t_min=t_min,
        t_max=t_max,
        indexes_to_remove=[
            "care_site_id",
            "care_site_short_name",
            "specialties_set",
            "care_site_specialty",
            "stay_type",
        ],
    )
    visit_probe.predictor["EHR_functionality"] = "Intensive care stays"
    data.append(
        visit_probe.predictor.merge(visit_model.estimates, on=visit_probe._index)
    )
    visit_probe.reset_predictor()
    visit_model.reset_estimates()

    model_goodness_fit_chart = reduce(
        lambda chart_1, chart_2: alt.vconcat(chart_1, chart_2, spacing=5),
        charts.values(),
    )
    model_goodness_fit_chart = (
        model_goodness_fit_chart.properties(
            title="Normalized completeness estimate = c(Δt) / c₀"
        )
        .configure_legend(
            labelLimit=500,
            orient="none",
            legendX=720,
            legendY=-100,
            labelFontSize=19,
            titleFontSize=20,
            symbolSize=300,
        )
        .configure_axis(
            labelFontSize=19,
            titleFontSize=20,
            labelLimit=500,
        )
        .configure_title(
            fontSize=20,
            dx=-10,
            orient="left",
            align="center",
            anchor="middle",
        )
    )
    model_goodness_fit_data = pd.concat(data)[
        [
            "EHR_functionality",
            "care_site_level",
            "care_site_id",
            "care_site_short_name",
            "date",
            "c",
            "c_0",
            "t_0",
            "error",
        ]
    ]
    model_goodness_fit_data["normalized_date"] = month_diff(
        model_goodness_fit_data["date"], model_goodness_fit_data["t_0"]
    ).astype(int)
    model_goodness_fit_data["normalized_c"] = model_goodness_fit_data["c"].where(
        (model_goodness_fit_data["normalized_date"] < 0)
        | (model_goodness_fit_data["c_0"] == 0),
        model_goodness_fit_data["c"] / model_goodness_fit_data["c_0"],
    )
    model_goodness_fit_data["model"] = 1
    model_goodness_fit_data["model"] = model_goodness_fit_data["model"].where(
        model_goodness_fit_data["normalized_date"] >= 0, 0
    )
    return model_goodness_fit_chart, model_goodness_fit_data


def get_normalized_line_config(with_legend: bool = True):
    probe_line_config = dict(
        legend_title="Normalized completeness estimate (mean)",
        encode=dict(
            strokeDash=alt.StrokeDash(
                "legend_predictor",
                title="",
                legend=alt.Legend(
                    symbolType="stroke",
                    symbolStrokeColor="steelblue",
                    symbolSize=500,
                    legendX=20,
                    legendY=-60,
                    orient="none",
                    labelFontSize=20,
                    symbolStrokeWidth=3,
                )
                if with_legend
                else None,
            )
        ),
    )
    error_line_config = dict(
        legend_title="Normalized completeness estimate (standard deviation)",
        mark_errorband=dict(extent="stdev", clip=True),
        encode=dict(
            stroke=alt.Stroke(
                "legend_error_band",
                title="",
                legend=alt.Legend(
                    symbolType="square",
                    symbolSize=300,
                    legendX=20,
                    legendY=-25,
                    orient="none",
                    labelFontSize=20,
                )
                if with_legend
                else None,
            ),
        ),
    )
    model_line_config = dict(
        legend_title="Model",
        mark_line=dict(
            color="black",
            interpolate="step-after",
            strokeDash=[5, 5],
        ),
        encode=dict(
            y="model:Q",
            strokeWidth=alt.StrokeWidth(
                field="legend_model",
                title="",
                legend=alt.Legend(
                    symbolType="stroke",
                    symbolStrokeColor="steelblue",
                    symbolSize=500,
                    legendX=500,
                    legendY=-60,
                    orient="none",
                    labelFontSize=20,
                    symbolStrokeWidth=3,
                    symbolDash=[5, 5],
                )
                if with_legend
                else None,
            ),
        ),
    )
    return probe_line_config, error_line_config, model_line_config


def get_main_chart_config(with_X_title: bool, Y_title: str):
    return dict(
        encode=dict(
            x=alt.X(
                "normalized_date:Q",
                title="Δt = (t - t₀) months" if with_X_title else "",
                axis=alt.Axis(labels=with_X_title),
                scale=alt.Scale(nice=False),
            ),
            y=alt.Y(
                "mean(normalized_c):Q",
                title=Y_title,
                axis=alt.Axis(grid=True),
                scale=alt.Scale(domainMin=0),
            ),
            color=alt.Color(
                "value:N",
                sort={"field": "n_visit", "op": "sum", "order": "descending"},
                title="Care site level",
            ),
        ),
        properties=dict(
            height=250,
            width=900,
        ),
    )


def get_selections(min_c_0: float):
    c_0_min_slider = alt.binding_range(
        min=0,
        max=1,
        step=0.01,
        name="c₀ min: ",
    )
    c_0_min_selection = alt.selection_point(
        name="c_0_min",
        fields=["c_0_min"],
        bind=c_0_min_slider,
        value=min_c_0,
    )
    estimates_selections = [c_0_min_selection]
    estimates_filters = [alt.datum.c_0 >= c_0_min_selection.c_0_min]
    return estimates_selections, estimates_filters
