import pandas as pd
from edsteva.models.rectangle_function import RectangleFunction
from edsteva.models.step_function import StepFunction
from IPython.display import display

from cse_210033 import BASE_DIR
from cse_210033.statistical_analysis.utils.supplementary_variables import t_test


def ehr_summary_table():
    summary_table = pd.DataFrame(
        data={
            "Average error (std)": [],
            "Average t0 (std)": [],
            "Average t1 (std)": [],
            "Average c0 (std)": [],
        }
    )
    ehr_modeling_folder_path = BASE_DIR / "data" / "ehr_modeling"
    models_folder_path = ehr_modeling_folder_path / "models"
    visit_model_path = models_folder_path / "visit.pickle"
    condition_model_path = models_folder_path / "condition.pickle"
    condition_model_per_visit_path = models_folder_path / "condition_per_visit.pickle"
    note_model_path = models_folder_path / "note.pickle"
    note_model_per_visit_path = models_folder_path / "note_per_visit.pickle"
    icu_model_path = models_folder_path / "icu_rectangle.pickle"

    visit_model = StepFunction()
    visit_model.load(visit_model_path)

    condition_model = StepFunction()
    condition_model.load(condition_model_path)
    condition_model_per_visit = StepFunction()
    condition_model_per_visit.load(condition_model_per_visit_path)

    note_model = StepFunction()
    note_model.load(note_model_path)
    note_model_per_visit = StepFunction()
    note_model_per_visit.load(note_model_per_visit_path)

    icu_model = RectangleFunction()
    icu_model.load(icu_model_path)
    model_total = []

    # Filter models
    model_hospit = visit_model.estimates[
        (visit_model.estimates.care_site_level == "Hôpital")
        & (visit_model.estimates.stay_type == "hospitalisés")
    ].copy()
    model_total.append(model_hospit)
    model_emergency = visit_model.estimates[
        (visit_model.estimates.care_site_level == "Hôpital")
        & (visit_model.estimates.stay_type == "urgence")
    ].copy()
    model_total.append(model_emergency)
    model_consultation = note_model.estimates[
        (note_model.estimates.care_site_level == "Unité Fonctionnelle (UF)")
        & (note_model.estimates.stay_type == "All")
        & (note_model.estimates.note_type == "consultation")
    ].copy()
    model_total.append(model_consultation)
    model_prescription = note_model_per_visit.estimates[
        (note_model_per_visit.estimates.care_site_level == "Unité Fonctionnelle (UF)")
        & (note_model_per_visit.estimates.stay_type == "hospit")
        & (note_model_per_visit.estimates.note_type == "prescription")
    ].copy()
    model_total.append(model_prescription)
    model_diagnosis = condition_model_per_visit.estimates[
        (
            condition_model_per_visit.estimates.care_site_level
            == "Unité Fonctionnelle (UF)"
        )
        & (condition_model_per_visit.estimates.source_system == "ORBIS")
        & (condition_model_per_visit.estimates.diag_type == "DP_DR")
        & (condition_model_per_visit.estimates.condition_type == "All")
    ].copy()
    model_total.append(model_diagnosis)
    model_icu = visit_model.estimates[
        (visit_model.estimates.care_site_level == "Unité d’hébergement (UH)")
        & (visit_model.estimates.specialties_set == "ICU")
    ].copy()
    model_total.append(model_icu)
    model_icu_rectangle = icu_model.estimates[
        (icu_model.estimates.care_site_level == "Unité d’hébergement (UH)")
        & (icu_model.estimates.specialties_set == "ICU")
    ].copy()
    model_total.append(model_icu_rectangle)
    ehr_models = dict(
        Hospitalization=model_hospit,
        Emergency=model_emergency,
        Consultation=model_consultation,
        Prescription=model_prescription,
        Diagnosis=model_diagnosis,
        ICU=model_icu,
        ICU_rectangle=model_icu_rectangle,
        Total=pd.concat(model_total),
    )
    for index, model in ehr_models.items():
        error_model = (
            str(round(model.error.mean(), 4))
            + " ("
            + str(round(model.error.std(), 4))
            + ")"
        )
        t0_model = (
            str(model.t_0.mean().date()) + " (" + str(model.t_0.std().days) + " days)"
        )
        t1_model = (
            (str(model.t_1.mean().date()) + " (" + str(model.t_1.std().days) + " days)")
            if "t_1" in model.columns
            else None
        )
        c0_model = (
            str(round(model.c_0.mean(), 4))
            + " ("
            + str(round(model.c_0.std(), 4))
            + ")"
        )
        summary_table.loc[index] = [error_model, t0_model, t1_model, c0_model]
    return summary_table


def cohort_summary_table():
    summary_table = pd.DataFrame(
        data={
            "Number of stays": [],
            "Number of patients": [],
            "Number of hospitals": [],
            "Number of departments": [],
            "Number of units": [],
            "Average age at stay (std)": [],
        }
    )
    cohort_path = BASE_DIR / "data" / "cohort_selection"
    cohort_tables = dict(
        Cohort=pd.read_pickle(cohort_path / "cohort_visit.pickle"),
        QI1_Hospitalization=pd.read_pickle(cohort_path / "hospit_visit.pickle"),
        QI2_Emergency=pd.read_pickle(cohort_path / "emergency_visit.pickle"),
        QI3_Consultation=pd.read_pickle(cohort_path / "consultation_note.pickle"),
        QI4_Prescription=pd.read_pickle(cohort_path / "prescription_note.pickle"),
        QI5_ICU=pd.read_pickle(cohort_path / "icu_visit.pickle"),
        EI1_Bronchiolitis=pd.read_pickle(
            cohort_path / "bronchiolitis_condition.pickle"
        ),
        EI2_Flu=pd.read_pickle(cohort_path / "flu_condition.pickle"),
        EI3_Gastroenteritis=pd.read_pickle(
            cohort_path / "gastroenteritis_condition.pickle"
        ),
        EI4_Nasopharyngitis=pd.read_pickle(
            cohort_path / "nasopharyngitis_condition.pickle"
        ),
    )

    for index, table in cohort_tables.items():
        if "visit_occurrence_id" in table.columns:
            stays = str(table.visit_occurrence_id.nunique())
        elif "visit_cohort_id" in table.columns:
            stays = str(table.visit_cohort_id.nunique())
        else:
            stays = None
        patients = (
            str(table.person_id.nunique()) if "person_id" in table.columns else None
        )
        hospit = (
            str(table.care_site_id.nunique())
            if "care_site_id" in table.columns
            else None
        )
        departments = (
            str(
                table[
                    table.detail_care_site_level == "Unité Fonctionnelle (UF)"
                ].detail_care_site_id.nunique()
            )
            if "detail_care_site_id" in table.columns
            else None
        )
        units = (
            str(
                table[
                    table.detail_care_site_level == "Unité d’hébergement (UH)"
                ].detail_care_site_id.nunique()
            )
            if "detail_care_site_id" in table.columns
            else None
        )
        age = (
            (
                str(round(table.age_at_stay.mean(), 1))
                + " ("
                + str(round(table.age_at_stay.std(), 1))
                + ")"
            )
            if "age_at_stay" in table.columns
            else None
        )

        summary_table.loc[index] = [stays, patients, hospit, departments, units, age]
    return summary_table


def stats_analysis_summary_tables(
    stats_config,
    min_c_0: float = 0.0,
    max_error: float = 1.0,
    threshold: str = "before_30_days",
):
    statistical_analysis_path = BASE_DIR / "data" / "statistical_analysis"
    cs_count_summary = pd.read_pickle(
        statistical_analysis_path / "cs_count_summary.pkl"
    )
    cs_count_summary = cs_count_summary[
        (cs_count_summary.min_c_0 == min_c_0)
        & (cs_count_summary.max_error == max_error)
    ]
    cso_analysis = cs_count_summary.copy()
    cso_analysis["Statistical analysis"] = "Complete-source-only analysis"
    cso_analysis["cs_considered"] = (
        cso_analysis["cso_cs_count"].astype(str)
        + " ("
        + (cso_analysis["cso_cs_count"] / cso_analysis["naive_cs_count"] * 100).map(
            "{:,.1f}".format
        )
        + " %)"
    )
    naive_analysis = cs_count_summary.copy()
    naive_analysis["Statistical analysis"] = "Naive analysis"
    naive_analysis["cs_considered"] = (
        naive_analysis["naive_cs_count"].astype(str) + " (100 %)"
    )
    cs_count_summary = pd.concat([naive_analysis, cso_analysis])[
        [
            "start_observation_date",
            "Statistical analysis",
            "cs_considered",
            "outcome_name",
        ]
    ]
    outcomes = [
        "hospit_visit",
        "emergency_visit",
        "consultation_note",
        "prescription_note",
        "icu_visit",
        "icu_visit_rectangle",
        "bronchiolitis_condition",
        "flu_condition",
    ]
    summary_tables = []
    for outcome in outcomes:
        outcome_indicator = pd.read_pickle(
            statistical_analysis_path / "{}.pkl".format(outcome)
        )
        outcome_indicator["outcome_name"] = outcome
        if "rate" in outcome_indicator.columns:
            summary_table = t_test(
                outcome_indicator,
                x_col="sub_cohort",
                y_col="rate",
                as_string=True,
            )
            summary_table = summary_table[
                (summary_table.care_site_id == "All")
                & (summary_table.min_c_0 == min_c_0)
                & (summary_table.max_error == max_error)
            ]
            if "threshold" in summary_table.columns:
                summary_table = summary_table[summary_table.threshold == threshold]
            summary_table = summary_table.groupby(
                ["start_observation_date", "Statistical analysis"]
            ).agg({"alpha_0": "first", "alpha_1": "first", "outcome_name": "first"})
        else:
            summary_table = outcome_indicator.groupby(
                ["start_observation_date", "Statistical analysis"]
            ).agg({"outcome_name": "first"})
        summary_table = summary_table.merge(
            cs_count_summary,
            on=[
                "start_observation_date",
                "Statistical analysis",
                "outcome_name",
            ],
        )
        summary_table = summary_table.replace(
            {
                "Statistical analysis": {
                    "Naive analysis": "0 - Naive analysis",
                    "Complete-source-only analysis": "1 - Complete-source-only analysis",
                }
            },
        )
        summary_table = summary_table.groupby(
            ["start_observation_date", "Statistical analysis"]
        ).first()
        summary_table = summary_table.rename(
            index={
                "0 - Naive analysis": "Naive analysis",
                "1 - Complete-source-only analysis": "Complete-source-only analysis",
            }
        )
        print("\033[1m" + stats_config[outcome]["event_name"])
        if "alpha_0" in summary_table.columns:
            display(summary_table[["cs_considered", "alpha_0", "alpha_1"]])
        else:
            display(summary_table[["cs_considered"]])
        summary_tables.append(summary_table)
    return pd.concat(summary_tables)
