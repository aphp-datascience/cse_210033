import altair as alt
import pandas as pd


def plot_ehr_context(
    visit_predictor: pd.DataFrame,
    condition_predictor: pd.DataFrame,
    note_predictor: pd.DataFrame,
    start_date: str,
):
    hospit_predictor = (
        visit_predictor[
            (visit_predictor.stay_type == "hospitalisés")
            & (visit_predictor.care_site_level == "Hôpital")
            & (visit_predictor.specialties_set == "All")
            & (visit_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_visit": "sum"})
    ).rename(columns={"n_visit": "n_event"})
    hospit_predictor["max_event"] = hospit_predictor.n_event.max()
    hospit_predictor["c"] = hospit_predictor.n_event / hospit_predictor.max_event
    hospit_predictor["event_type"] = "1. Hospitalization records"

    emergency_predictor = (
        visit_predictor[
            (visit_predictor.stay_type == "urgence")
            & (visit_predictor.care_site_level == "Hôpital")
            & (visit_predictor.specialties_set == "All")
            & (visit_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_visit": "sum"})
    ).rename(columns={"n_visit": "n_event"})
    emergency_predictor["max_event"] = emergency_predictor.n_event.max()
    emergency_predictor["c"] = (
        emergency_predictor.n_event / emergency_predictor.max_event
    )
    emergency_predictor["event_type"] = "2. Emergency department records"

    note_consultation_predictor = (
        note_predictor[
            (note_predictor.note_type == "consultation")
            & (note_predictor.care_site_level == "Hôpital")
            & (note_predictor.stay_type == "All")
            & (note_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_note": "sum"})
    ).rename(columns={"n_note": "n_event"})
    note_consultation_predictor["max_event"] = note_consultation_predictor.n_event.max()
    note_consultation_predictor["c"] = (
        note_consultation_predictor.n_event / note_consultation_predictor.max_event
    )
    note_consultation_predictor["event_type"] = "3. Consultation reports"

    note_prescription_predictor = (
        note_predictor[
            (note_predictor.note_type == "prescription")
            & (note_predictor.care_site_level == "Hôpital")
            & (note_predictor.stay_type == "All")
            & (note_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_note": "sum"})
    ).rename(columns={"n_note": "n_event"})
    note_prescription_predictor["max_event"] = note_prescription_predictor.n_event.max()
    note_prescription_predictor["c"] = (
        note_prescription_predictor.n_event / note_prescription_predictor.max_event
    )
    note_prescription_predictor["event_type"] = "4. Prescription reports"

    condition_ORBIS_predictor = (
        condition_predictor[
            (condition_predictor.source_system == "ORBIS")
            & (condition_predictor.care_site_level == "Hôpital")
            & (condition_predictor.stay_type == "hospit")
            & (condition_predictor.condition_type == "All")
            & (condition_predictor.diag_type == "DP_DR")
            & (condition_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_condition": "sum"})
    ).rename(columns={"n_condition": "n_event"})
    condition_ORBIS_predictor["max_event"] = condition_ORBIS_predictor.n_event.max()
    condition_ORBIS_predictor["c"] = (
        condition_ORBIS_predictor.n_event / condition_ORBIS_predictor.max_event
    )
    condition_ORBIS_predictor["event_type"] = "5. Diagnoses"

    icu_predictor = (
        visit_predictor[
            (visit_predictor.specialties_set == "ICU")
            & (visit_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_visit": "sum"})
    ).rename(columns={"n_visit": "n_event"})
    icu_predictor["max_event"] = icu_predictor.n_event.max()
    icu_predictor["c"] = icu_predictor.n_event / icu_predictor.max_event
    icu_predictor["event_type"] = "6. Intensive care unit records"

    condition_AREM_predictor = (
        condition_predictor[
            (condition_predictor.source_system == "AREM")
            & (condition_predictor.care_site_level == "Hôpital")
            & (condition_predictor.stay_type == "hospit")
            & (condition_predictor.condition_type == "All")
            & (condition_predictor.diag_type == "DP_DR")
            & (condition_predictor.date >= start_date)
        ]
        .groupby(["date"], as_index=False)
        .agg({"n_condition": "sum"})
    ).rename(columns={"n_condition": "n_event"})
    condition_AREM_predictor["max_event"] = condition_AREM_predictor.n_event.max()
    condition_AREM_predictor["c"] = (
        condition_AREM_predictor.n_event / condition_AREM_predictor.max_event
    )
    condition_AREM_predictor["event_type"] = "7. Curated claims diagnosis"
    ehr_context_data = pd.concat(
        [
            condition_ORBIS_predictor,
            # condition_AREM_predictor,
            note_consultation_predictor,
            note_prescription_predictor,
            emergency_predictor,
            hospit_predictor,
            icu_predictor,
        ]
    )

    ehr_context_chart = (
        (
            alt.Chart(ehr_context_data)
            .mark_line()
            .encode(
                color=alt.Color("event_type", title="EHR functionality"),
                strokeDash=alt.StrokeDash(
                    "event_type",
                    scale=alt.Scale(
                        range=[
                            [0, 0],
                            [15, 7],
                            [10, 5],
                            [5, 3],
                            [3, 1],
                            [5, 3, 1, 3],
                        ]
                    ),
                ),
                x=alt.X(
                    "date:T",
                    title="Collection date",
                    axis=alt.Axis(
                        tickCount="year", labelAngle=0, grid=True, format="%Y"
                    ),
                ),
                y=alt.Y(
                    "c:Q",
                    title="Normalized No. of data points",
                ),
            )
        )
        .properties(width=700)
        .configure_axis(
            labelFontSize=17,
            titleFontSize=18,
        )
        .configure_axisX(
            labelFlush=False,
        )
        .configure_legend(
            titleFontSize=18,
            labelFontSize=17,
            title=None,
            symbolSize=1200,
            symbolStrokeWidth=3,
            labelLimit=500,
        )
        .configure_view(strokeWidth=0)
    )
    return ehr_context_chart, ehr_context_data
