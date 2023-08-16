from typing import Dict, List

from edsteva.utils.framework import is_koalas
from edsteva.utils.typing import Data, DataFrame

from .utils.add_events import add_patient_info
from .utils.filter_events import (
    clean_date,
    filter_date,
    filter_event,
    filter_first_event,
    filter_note_type,
)
from .utils.prepare_df import (
    prepare_care_site,
    prepare_condition_occurrence,
    prepare_ghm,
    prepare_note,
    prepare_note_care_site,
    prepare_person,
    prepare_visit_detail,
    prepare_visit_occurrence,
)


def cohort_selection(
    data: Data,
    prod_data: Data,
    AREM_data: Data,
    start_date: str,
    end_date: str,
    note_source_system: str,
    person_source_system: str,
    visit_source_system: str,
    ghm_source_system: str,
    condition_source_system: str,
    cmd: Dict[str, List[str]],
    hospit_stay_type_regex: str,
    hospit_stay_source_regex: str,
    emergency_stay_type_regex: str,
    emergency_stay_source_regex: str,
    consultation_note_type_regex: str,
    prescription_note_type_regex: str,
    condition_transfer_type_regex: str,
    diag_regex: List[str],
    bronchiolite_regex: str,
    flu_regex: str,
    gastroenteritis_regex: str,
    nasopharyngitis_regex: str,
    icu_regex: str,
) -> Dict[str, DataFrame]:
    outcomes = {}
    # Table Extraction
    care_site = prepare_care_site(data=data)
    visit_occurrence = prepare_visit_occurrence(
        data=data, visit_source_system=visit_source_system
    )
    ghm = prepare_ghm(AREM_data=AREM_data, cmd_source_system=ghm_source_system, cmd=cmd)
    person = prepare_person(data=data, person_source_system=person_source_system)
    visit_detail = prepare_visit_detail(data=data, care_site=care_site)
    condition_occurrence = prepare_condition_occurrence(
        data=data,
        AREM_data=AREM_data,
        visit_occurrence=visit_occurrence,
        condition_source_system=condition_source_system,
        diag_regex=diag_regex,
    )
    note = prepare_note(data=data, note_source_system=note_source_system)
    note_care_site = prepare_note_care_site(extra_data=prod_data, care_site=care_site)
    condition_detail = visit_detail[
        visit_detail.transfer_type == condition_transfer_type_regex
    ].drop(columns="visit_occurrence_id")
    # Cohort visit
    cohort_visit = filter_event(
        df=visit_occurrence,
        col_to_filter="stay_type",
        event_regex=hospit_stay_type_regex,
    )
    cohort_visit = filter_event(
        df=cohort_visit,
        col_to_filter="stay_source",
        event_regex=hospit_stay_source_regex,
    )
    cohort_visit = cohort_visit.merge(ghm, on="visit_occurrence_id", how="left").rename(
        columns={
            "visit_start_datetime": "cohort_stay_start",
            "visit_end_datetime": "cohort_stay_end",
            "visit_occurrence_id": "visit_cohort_id",
        }
    )
    cohort_visit = filter_date(
        cohort_visit=cohort_visit, start_date=start_date, end_date=end_date
    )
    cohort_visit = add_patient_info(
        visit=cohort_visit, person=person
    )  # Filter also patients opposed to research
    if is_koalas(cohort_visit):
        cohort_visit = cohort_visit.spark.cache()
    # Store result
    outcomes["cohort_visit"] = cohort_visit

    # Outcome 1 : hospit visit after cohort stay
    # Filter event
    hospit_visit = filter_event(
        df=visit_occurrence,
        col_to_filter="stay_type",
        event_regex=hospit_stay_type_regex,
    )
    hospit_visit = filter_event(
        df=hospit_visit,
        col_to_filter="stay_source",
        event_regex=hospit_stay_source_regex,
    )
    # Filter on cohort patients
    hospit_visit = hospit_visit[
        [
            "visit_occurrence_id",
            "person_id",
            "care_site_id",
            "visit_start_datetime",
        ]
    ].merge(cohort_visit[["person_id"]].drop_duplicates(), on="person_id")
    # Store result
    outcomes["hospit_visit"] = hospit_visit

    # Outcome 2 : emergency visit after cohort stay
    # Filter event
    emergency_visit = filter_event(
        df=visit_occurrence,
        col_to_filter="stay_type",
        event_regex=emergency_stay_type_regex,
    )
    emergency_visit = filter_event(
        df=emergency_visit,
        col_to_filter="stay_source",
        event_regex=emergency_stay_source_regex,
    )
    # Filter on cohort patients
    emergency_visit = emergency_visit[
        [
            "visit_occurrence_id",
            "person_id",
            "care_site_id",
            "visit_start_datetime",
        ]
    ].merge(cohort_visit[["person_id"]].drop_duplicates(), on="person_id")
    # Store result
    outcomes["emergency_visit"] = emergency_visit

    # Outcome 3: consultation doc after cohort stay
    # Filter event
    consultation_note = filter_note_type(
        note=note, note_type_regex=consultation_note_type_regex
    )
    # Keep one note per visit
    consultation_note = filter_first_event(
        df=consultation_note, col_date="note_datetime"
    )
    # Clean date
    consultation_note = clean_date(df=consultation_note, col_date="note_datetime")
    # Filter on cohort patients
    consultation_note = consultation_note[
        [
            "person_id",
            "visit_occurrence_id",
            "note_datetime",
            "note_id",
            "note_type",
        ]
    ].merge(cohort_visit[["person_id"]].drop_duplicates(), on="person_id")
    # Add detail_care_site_id
    consultation_note = consultation_note.merge(
        note_care_site,
        on="note_id",
    ).drop(columns="note_id")
    # Add care_site_id
    consultation_note = consultation_note.merge(
        visit_occurrence[
            [
                "visit_occurrence_id",
                "care_site_id",
                "visit_start_datetime",
            ]
        ],
        on="visit_occurrence_id",
        how="left",
    )
    # Store result
    outcomes["consultation_note"] = consultation_note

    # Outcome 4: prescription doc during hospitalization
    # Filter event
    prescription_note = filter_note_type(
        note=note, note_type_regex=prescription_note_type_regex
    )
    # Keep one note per visit
    prescription_note = filter_first_event(
        df=prescription_note, col_date="note_datetime"
    )
    # Clean date
    prescription_note = clean_date(df=prescription_note, col_date="note_datetime")
    # Filter on cohort visits
    prescription_note = (
        cohort_visit[
            [
                "visit_cohort_id",
                "care_site_id",
                "cohort_stay_start",
            ]
        ]
        .rename(
            columns={
                "visit_cohort_id": "visit_occurrence_id",
            }
        )
        .drop_duplicates()
        .merge(
            prescription_note[
                [
                    "person_id",
                    "visit_occurrence_id",
                    "note_datetime",
                    "note_type",
                    "note_id",
                ]
            ],
            on="visit_occurrence_id",
        )
    )
    # Add detail_care_site_id
    prescription_note = prescription_note.merge(
        note_care_site,
        on="note_id",
    ).drop(columns="note_id")
    # Store result
    outcomes["prescription_note"] = prescription_note

    # Outcome 5: icu visit after cohort stay
    # Filter event
    icu_care_site = filter_event(
        df=care_site,
        col_to_filter="service_type",
        event_regex=icu_regex,
    )[["care_site_id", "service_type"]].rename(
        columns={
            "care_site_id": "detail_care_site_id",
        }
    )
    icu_visit = (
        visit_occurrence[
            [
                "visit_occurrence_id",
                "person_id",
                "care_site_id",
                "visit_start_datetime",
                "stay_type",
            ]
        ]
        .merge(
            visit_detail,
            on="visit_occurrence_id",
            how="inner",
        )
        .merge(
            icu_care_site,
            on="detail_care_site_id",
            how="inner",
        )
    )
    # Filter on cohort patients
    icu_visit = icu_visit.merge(
        cohort_visit[["person_id"]].drop_duplicates(), on="person_id"
    )
    # Keep one ICU per visit
    icu_visit = filter_first_event(df=icu_visit, col_date="visit_detail_start_datetime")
    # Store result
    outcomes["icu_visit"] = icu_visit

    # Outcome 6: bronchiolitis condition during hospitalization
    # Filter event
    bronchiolitis_condition = filter_event(
        df=condition_occurrence,
        col_to_filter="condition_source_value",
        event_regex=bronchiolite_regex,
    )
    # Keep one condition per visit
    bronchiolitis_condition = filter_first_event(
        df=bronchiolitis_condition, col_date="condition_start_datetime"
    )
    # Filter on cohort visits
    bronchiolitis_condition = (
        cohort_visit[
            [
                "person_id",
                "visit_cohort_id",
                "care_site_id",
                "cohort_stay_start",
            ]
        ]
        .rename(
            columns={
                "visit_cohort_id": "visit_occurrence_id",
            }
        )
        .drop_duplicates()
        .merge(
            bronchiolitis_condition[
                [
                    "visit_occurrence_id",
                    "visit_detail_id",
                    "diag_type",
                    "condition_source_value",
                    "condition_start_datetime",
                ]
            ],
            on="visit_occurrence_id",
        )
    )
    # Add detail_care_site_id
    if condition_source_system == "ORBIS":
        bronchiolitis_condition = bronchiolitis_condition.merge(
            condition_detail,
            on="visit_detail_id",
            how="left",
        )
    # Store result
    outcomes["bronchiolitis_condition"] = bronchiolitis_condition

    # Outcome 7: flu condition during hospitalization
    # Filter event
    flu_condition = filter_event(
        df=condition_occurrence,
        col_to_filter="condition_source_value",
        event_regex=flu_regex,
    )
    # Keep one condition per visit
    flu_condition = filter_first_event(
        df=flu_condition, col_date="condition_start_datetime"
    )
    # Filter on cohort visits
    flu_condition = (
        cohort_visit[
            [
                "person_id",
                "visit_cohort_id",
                "care_site_id",
                "cohort_stay_start",
            ]
        ]
        .rename(
            columns={
                "visit_cohort_id": "visit_occurrence_id",
            }
        )
        .drop_duplicates()
        .merge(
            flu_condition[
                [
                    "visit_occurrence_id",
                    "visit_detail_id",
                    "diag_type",
                    "condition_source_value",
                    "condition_start_datetime",
                ]
            ],
            on="visit_occurrence_id",
        )
    )
    # Add detail_care_site_id
    if condition_source_system == "ORBIS":
        flu_condition = flu_condition.merge(
            condition_detail,
            on="visit_detail_id",
            how="left",
        )
    # Store result
    outcomes["flu_condition"] = flu_condition

    # Outcome 8: gastroenteritis condition during hospitalization
    # Filter event
    gastroenteritis_condition = filter_event(
        df=condition_occurrence,
        col_to_filter="condition_source_value",
        event_regex=gastroenteritis_regex,
    )
    # Keep one condition per visit
    gastroenteritis_condition = filter_first_event(
        df=gastroenteritis_condition, col_date="condition_start_datetime"
    )
    # Filter on cohort visits
    gastroenteritis_condition = (
        cohort_visit[
            [
                "person_id",
                "visit_cohort_id",
                "care_site_id",
                "cohort_stay_start",
            ]
        ]
        .rename(
            columns={
                "visit_cohort_id": "visit_occurrence_id",
            }
        )
        .drop_duplicates()
        .merge(
            gastroenteritis_condition[
                [
                    "visit_occurrence_id",
                    "visit_detail_id",
                    "diag_type",
                    "condition_source_value",
                    "condition_start_datetime",
                ]
            ],
            on="visit_occurrence_id",
        )
    )
    # Add detail_care_site_id
    if condition_source_system == "ORBIS":
        gastroenteritis_condition = gastroenteritis_condition.merge(
            condition_detail,
            on="visit_detail_id",
            how="left",
        )
    # Store result
    outcomes["gastroenteritis_condition"] = gastroenteritis_condition

    # Outcome 9: nasopharyngitis condition during hospitalization
    # Filter event
    nasopharyngitis_condition = filter_event(
        df=condition_occurrence,
        col_to_filter="condition_source_value",
        event_regex=nasopharyngitis_regex,
    )
    # Keep one condition per visit
    nasopharyngitis_condition = filter_first_event(
        df=nasopharyngitis_condition, col_date="condition_start_datetime"
    )
    # Filter on cohort visits
    nasopharyngitis_condition = (
        cohort_visit[
            [
                "person_id",
                "visit_cohort_id",
                "care_site_id",
                "cohort_stay_start",
            ]
        ]
        .rename(
            columns={
                "visit_cohort_id": "visit_occurrence_id",
            }
        )
        .drop_duplicates()
        .merge(
            nasopharyngitis_condition[
                [
                    "visit_occurrence_id",
                    "visit_detail_id",
                    "diag_type",
                    "condition_source_value",
                    "condition_start_datetime",
                ]
            ],
            on="visit_occurrence_id",
        )
    )
    # Add detail_care_site_id
    if condition_source_system == "ORBIS":
        nasopharyngitis_condition = nasopharyngitis_condition.merge(
            condition_detail,
            on="visit_detail_id",
            how="left",
        )
    # Store result
    outcomes["nasopharyngitis_condition"] = nasopharyngitis_condition

    return outcomes
