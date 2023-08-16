from typing import Dict

import altair as alt
import pandas as pd
import polars as pl
from edsteva.models.rectangle_function import RectangleFunction
from edsteva.models.step_function import StepFunction
from edsteva.probes import VisitProbe

from cse_210033.statistical_analysis.utils.complete_source import filter_estimates


def plot_ehr_rectangle_model(
    visit_probe: VisitProbe,
    icu_probe: VisitProbe,
    visit_model: StepFunction,
    icu_model: RectangleFunction,
    config: Dict,
):
    ehr_rectangle_models_data = []
    # Load parameters from config
    example_unit_id = config["ehr_modeling"]["example_unit_id"]
    stats_config = config["statistical_analysis"]
    end_date = config["ehr_modeling"]["end_date"]

    # ICU Visit
    visit_predictor = visit_probe.predictor.copy()
    visit_predictor = visit_predictor[
        visit_predictor.care_site_id.isin(example_unit_id)
    ]
    unit_names = visit_predictor.care_site_short_name.unique()
    for i in range(len(unit_names)):
        visit_predictor.care_site_short_name.replace(
            unit_names[i], "Unit {}".format(i + 1), inplace=True
        )
    visit_probe.predictor = visit_predictor
    visit_predictor = visit_model.predict(visit_probe).copy()
    visit_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(visit_predictor), **stats_config["icu_visit"]
    ).to_pandas()
    visit_predictor = visit_predictor[["date", "care_site_short_name", "c", "c_hat"]]
    visit_predictor["data_type"] = 0
    ehr_rectangle_models_data.append(visit_predictor)
    visit_probe.reset_predictor()

    # ICU Visit
    icu_predictor = icu_probe.predictor.copy()
    icu_predictor = icu_predictor[icu_predictor.care_site_id.isin(example_unit_id)]
    for i in range(len(unit_names)):
        icu_predictor.care_site_short_name.replace(
            unit_names[i], "Unit {}".format(i + 1), inplace=True
        )
    icu_probe.predictor = icu_predictor
    icu_predictor = icu_model.predict(icu_probe).copy()
    icu_predictor = filter_estimates(
        ehr_estimates=pl.from_pandas(icu_predictor),
        **stats_config["icu_visit_rectangle"]
    ).to_pandas()
    icu_predictor = icu_predictor[["date", "care_site_short_name", "c", "c_hat"]]
    icu_predictor = icu_predictor[icu_predictor["date"] <= end_date]
    icu_predictor["data_type"] = 1
    ehr_rectangle_models_data.append(icu_predictor)
    icu_probe.reset_predictor()

    ehr_rectangle_models_data = pd.concat(ehr_rectangle_models_data)
    ehr_rectangle_models_data["legend_predictor"] = "Estimate"
    ehr_rectangle_models_data["legend_model"] = "Model"
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
                    "Unit 1",
                    "Unit 2",
                ],
                scale=alt.Scale(
                    range=[
                        "#b279a2",
                        "#ff9da6",
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
    ehr_rectangle_model_chart = alt.layer(
        probe_line, model_line, data=ehr_rectangle_models_data
    ).facet(
        row=alt.Row(
            "data_type",
            title="Completeness",
            header=alt.Header(
                titleFontSize=19,
                labelFontSize=20,
                labelFontWeight="bold",
                labelExpr="datum.label == 0 ? 'Intensive care stays' : ['Intensive care stays', 'with rectangle model']",
            ),
        ),
        spacing=15,
    )
    ehr_rectangle_model_chart = (
        ehr_rectangle_model_chart.configure_axis(
            labelFontSize=19,
            titleFontSize=20,
        )
        .configure_axisX(
            labelFlush=False,
            gridCap="round",
        )
        .configure_legend(
            labelFontSize=19,
            symbolStrokeWidth=3,
            title=None,
            symbolSize=500,
        )
        .configure_view(strokeWidth=0)
    )
    return ehr_rectangle_model_chart, ehr_rectangle_models_data
