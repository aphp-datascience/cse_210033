from typing import Dict

import altair as alt
import pandas as pd
import polars as pl
from edsteva.models.step_function import StepFunction
from edsteva.probes import ConditionProbe, NoteProbe, VisitProbe

from cse_210033.statistical_analysis.utils.complete_source import filter_estimates


def plot_ehr_models(
    visit_probe: VisitProbe,
    note_probe: NoteProbe,
    note_probe_per_visit: NoteProbe,
    condition_probe_per_visit: ConditionProbe,
    visit_model: StepFunction,
    note_model: StepFunction,
    note_per_visit_model: StepFunction,
    condition_per_visit_model: StepFunction,
    config: Dict,
):
    ehr_models_data = []
    # Load parameters from config
    example_hospital_id = config["ehr_modeling"]["example_hospital_id"]
    example_department_id = config["ehr_modeling"]["example_department_id"]
    stats_config = config["statistical_analysis"]

    # Hospit Visit
    visit_predictor = visit_probe.predictor.copy()
    visit_predictor = visit_predictor[
        visit_predictor.care_site_id.isin(example_hospital_id)
    ]
    hospital_names = visit_predictor.care_site_short_name.unique()
    for i in range(len(hospital_names)):
        visit_predictor.care_site_short_name.replace(
            hospital_names[i], "Hospital {}".format(i + 1), inplace=True
        )
    visit_probe.predictor = visit_predictor
    visit_predictor = visit_model.predict(visit_probe).copy()
    visit_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(visit_predictor), **stats_config["hospit_visit"]
    ).to_pandas()
    visit_predictor = visit_predictor[["date", "care_site_short_name", "c", "c_hat"]]
    visit_predictor["data_type"] = "Hospitalizations"
    ehr_models_data.append(visit_predictor)
    visit_probe.reset_predictor()

    # Emergency Visit
    visit_predictor = visit_probe.predictor.copy()
    visit_predictor = visit_predictor[
        visit_predictor.care_site_id.isin(example_hospital_id)
    ]
    for i in range(len(hospital_names)):
        visit_predictor.care_site_short_name.replace(
            hospital_names[i], "Hospital {}".format(i + 1), inplace=True
        )
    visit_probe.predictor = visit_predictor
    visit_predictor = visit_model.predict(visit_probe).copy()
    visit_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(visit_predictor), **stats_config["emergency_visit"]
    ).to_pandas()
    visit_predictor = visit_predictor[["date", "care_site_short_name", "c", "c_hat"]]
    visit_predictor["data_type"] = "Emergency stays"
    ehr_models_data.append(visit_predictor)
    visit_probe.reset_predictor()

    # Consultation Note
    note_predictor = note_probe.predictor.copy()
    note_predictor = note_predictor[
        note_predictor.care_site_id.isin(example_department_id)
    ]
    department_names = note_predictor.care_site_short_name.unique()
    for i in range(len(department_names)):
        note_predictor.care_site_short_name.replace(
            department_names[i], "Department {}".format(i + 1), inplace=True
        )
    note_probe.predictor = note_predictor
    note_predictor = note_model.predict(note_probe).copy()
    note_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(note_predictor),
        **stats_config["consultation_note"]
    ).to_pandas()
    note_predictor = note_predictor[["date", "care_site_short_name", "c", "c_hat"]]
    note_predictor["data_type"] = "Consultation reports"
    ehr_models_data.append(note_predictor)
    note_probe.reset_predictor()

    # Prescription Note
    note_predictor = note_probe_per_visit.predictor.copy()
    note_predictor = note_predictor[
        note_predictor.care_site_id.isin(example_department_id)
    ]
    for i in range(len(department_names)):
        note_predictor.care_site_short_name.replace(
            department_names[i], "Department {}".format(i + 1), inplace=True
        )
    note_probe_per_visit.predictor = note_predictor
    note_predictor = note_per_visit_model.predict(note_probe_per_visit).copy()
    note_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(note_predictor),
        **stats_config["prescription_note"]
    ).to_pandas()
    note_predictor = note_predictor[["date", "care_site_short_name", "c", "c_hat"]]
    note_predictor["data_type"] = "Prescription reports"
    ehr_models_data.append(note_predictor)
    note_probe_per_visit.reset_predictor()

    # ORBIS Condition
    condition_predictor = condition_probe_per_visit.predictor.copy()
    condition_predictor = condition_predictor[
        condition_predictor.care_site_id.isin(example_department_id)
    ]
    for i in range(len(department_names)):
        condition_predictor.care_site_short_name.replace(
            department_names[i], "Department {}".format(i + 1), inplace=True
        )
    condition_probe_per_visit.predictor = condition_predictor
    condition_predictor = condition_per_visit_model.predict(
        condition_probe_per_visit
    ).copy()
    condition_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(condition_predictor),
        **stats_config["bronchiolitis_condition"]
    ).to_pandas()
    condition_predictor = condition_predictor[
        ["date", "care_site_short_name", "c", "c_hat"]
    ]
    condition_predictor["data_type"] = "Diagnostic codes"
    ehr_models_data.append(condition_predictor)
    condition_probe_per_visit.reset_predictor()

    ehr_models_data = pd.concat(ehr_models_data)
    ehr_models_data["legend_predictor"] = "Estimate"
    ehr_models_data["legend_model"] = "Model"
    base = (
        alt.Chart()
        .encode(
            x=alt.X(
                "date:T",
                title="Collection date",
                axis=alt.Axis(tickCount="year", labelAngle=0, grid=True, format="%Y"),
            ),
            color=alt.Color(
                "care_site_short_name:N",
                legend=alt.Legend(orient="top"),
                sort=[
                    "Hospital 1",
                    "Hospital 2",
                    "Department 1",
                    "Department 2",
                ],
                scale=alt.Scale(
                    range=[
                        "#e45756",
                        "#72b7b2",
                        "#54a24b",
                        "#eeca3b",
                    ]
                ),
            ),
        )
        .properties(width=900, height=200)
    )
    probe_line = base.mark_line().encode(
        y=alt.Y("c:Q", title=None),
        strokeDash=alt.StrokeDash(
            "legend_predictor",
            title="",
            legend=alt.Legend(
                symbolType="stroke",
                symbolStrokeColor="grey",
                orient="none",
                legendX=0,
                legendY=-75,
            ),
        ),
    )
    model_line = base.mark_line(
        interpolate="step-after",
        strokeDash=[5, 5],
    ).encode(
        y=alt.Y("c_hat:Q"),
        strokeWidth=alt.StrokeWidth(
            field="legend_model",
            title="",
            legend=alt.Legend(
                symbolType="stroke",
                symbolStrokeColor="grey",
                symbolDash=[4, 4],
                orient="none",
                legendX=125,
                legendY=-75,
            ),
        ),
    )
    ehr_model_chart = alt.layer(probe_line, model_line, data=ehr_models_data).facet(
        row=alt.Row(
            "data_type",
            title="Completeness",
            header=alt.Header(
                titleFontSize=20, labelFontSize=19, labelFontWeight="bold"
            ),
            sort=[
                "Hospitalizations",
                "Emergency stays",
                "Consultation reports",
                "Prescription reports",
                "Diagnostic codes",
            ],
        ),
        spacing=15,
    )
    ehr_model_chart = (
        ehr_model_chart.configure_axis(
            labelFontSize=19,
            titleFontSize=20,
        )
        .configure_axisX(
            labelFlush=False,
            gridCap="round",
        )
        .configure_legend(
            labelFontSize=19,
            title=None,
            symbolStrokeWidth=3,
            symbolSize=500,
        )
        .configure_view(strokeWidth=0)
    )
    return ehr_model_chart, ehr_models_data
